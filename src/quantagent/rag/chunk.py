"""rag/chunk.py -- item-aware chunking of an SEC filing (architecture.md
§4.7: "structure-aware chunking that respects filing item boundaries --
never splits across Item 1A / Item 7").

Pure text logic, no I/O: imports `contracts` + `bs4` only, never
`data/providers/edgar.py` directly (the caller in `rag/ingest.py` wires the
two together).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from bs4 import BeautifulSoup

# Soft targets: the greedy accumulator in `_chunk_section_text` checks
# length only after adding one more line/sentence-piece, so a chunk may run
# somewhat over `CHUNK_TARGET_CHARS` (typically by less than one sentence's
# worth) -- this is a target, not a hard cap.
CHUNK_TARGET_CHARS = 3500
CHUNK_OVERLAP_CHARS = 500
_UNKNOWN_ITEM = "UNKNOWN"

# Trailing separator class covers a plain hyphen plus en-dash/em-dash
# (U+2013, U+2014) -- those two built from code points rather than pasted
# literally, so ruff's ambiguous-unicode check (RUF001) doesn't flag the
# source file.
_DASH_CHARS = chr(0x2013) + chr(0x2014)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_ITEM_HEADING_PATTERN = re.compile(
    r"^\s*item\s+(\d{1,2}[a-c]?)\.?\s*[-" + _DASH_CHARS + r":]?\s*", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    ticker: str
    cik: str
    form_type: str
    filed_at: date
    period: date | None
    item: str
    section_path: str
    chunk_id: str  # f"{accession_no}#{item}#{chunk_index:04d}" -- this IS the Evidence.ref


@dataclass(frozen=True, slots=True)
class Chunk:
    metadata: ChunkMetadata
    text: str


@dataclass(frozen=True, slots=True)
class FilingIdentity:
    """The subset of `data.providers.edgar.FilingRef` that chunking needs,
    decoupled so this module has no dependency on `data/` (rag/chunk.py
    must stay pure, importable by anything without pulling in httpx/redis).
    """

    ticker: str
    cik: str
    accession_no: str
    form_type: str
    filed_at: date
    period_of_report: date | None


def chunk_filing(html: str, identity: FilingIdentity) -> list[Chunk]:
    """Extracts plain text, finds item-section boundaries, and chunks each
    section independently -- the per-section chunking loop is what
    structurally guarantees a chunk never crosses an item boundary: overlap
    text only ever comes from the same section.
    """
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    lines = text.splitlines()
    boundaries = _find_item_boundaries(lines)
    sections = _split_into_sections(lines, boundaries)

    chunks: list[Chunk] = []
    for item_token, section_text in sections:
        section_path = f"{identity.form_type}#Item {item_token}"
        section_chunks = _chunk_section_text(
            section_text, target_chars=CHUNK_TARGET_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS
        )
        for index, chunk_text in enumerate(section_chunks):
            chunks.append(
                Chunk(
                    metadata=ChunkMetadata(
                        ticker=identity.ticker,
                        cik=identity.cik,
                        form_type=identity.form_type,
                        filed_at=identity.filed_at,
                        period=identity.period_of_report,
                        item=item_token,
                        section_path=section_path,
                        chunk_id=f"{identity.accession_no}#{item_token}#{index:04d}",
                    ),
                    text=chunk_text,
                )
            )
    return chunks


def _find_item_boundaries(lines: list[str]) -> list[tuple[int, str]]:
    """Returns `(line_index, item_token)` for the LAST occurrence of each
    distinct item token, sorted by document position.

    EDGAR filings almost always list every item twice: once in a Table of
    Contents near the top, once as the real section heading later. Taking
    the last occurrence is a simple, robust way to prefer the real heading
    over the TOC reference without full layout/font analysis -- a
    documented simplification, not a claim that every possible filing
    layout is handled.
    """
    last_seen: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = _ITEM_HEADING_PATTERN.match(line)
        if match:
            last_seen[match.group(1).upper()] = index
    return sorted(((line_index, token) for token, line_index in last_seen.items()))


def _split_into_sections(
    lines: list[str], boundaries: list[tuple[int, str]]
) -> list[tuple[str, str]]:
    """Returns `(item_token, section_text)` pairs. A filing with zero
    recognisable item headings degrades to one `UNKNOWN` section covering
    the whole document -- surfaced by the caller as a limitation (I8),
    never silently misattributed to a real item.
    """
    if not boundaries:
        return [(_UNKNOWN_ITEM, "\n".join(lines))]

    sections: list[tuple[str, str]] = []
    for i, (start_index, token) in enumerate(boundaries):
        end_index = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        sections.append((token, "\n".join(lines[start_index:end_index])))
    return sections


def _chunk_section_text(text: str, *, target_chars: int, overlap_chars: int) -> list[str]:
    """Greedily accumulates paragraphs up to `target_chars`, then starts the
    next chunk with the trailing `overlap_chars` of the previous chunk
    prepended. Operates on one section's text alone, so overlap never
    crosses an item boundary.

    Accumulates by individual line, not blank-line-separated paragraphs:
    `BeautifulSoup.get_text(separator="\\n")` emits one line per source
    text node with no blank line between them (verified empirically -- a
    `\\n\\s*\\n` paragraph split would never match, silently degrading every
    section into a single oversized chunk regardless of `target_chars`).
    A single line longer than `target_chars` on its own (a whole paragraph
    of prose in one text node, no internal `<br>`, which real filing HTML
    does produce) is pre-split by `_split_oversized_line` before
    accumulation, so the accumulator never has to accept an over-length
    candidate just because `current` was still empty.
    """
    raw_lines = [line for line in re.split(r"\n+", text) if line.strip()]
    lines = [
        piece
        for raw_line in raw_lines
        for piece in _split_oversized_line(raw_line, target_chars=target_chars)
    ]

    chunks: list[str] = []
    current = ""
    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > target_chars and current:
            chunks.append(current)
            overlap_tail = current[-overlap_chars:] if len(current) > overlap_chars else current
            current = f"{overlap_tail}\n{line}"
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_oversized_line(line: str, *, target_chars: int) -> list[str]:
    """Sub-splits a single over-length line on sentence boundaries; a
    pathological run-on line with no sentence punctuation at all is
    hard-sliced by character count as a last resort so no piece this
    function returns can itself still exceed `target_chars`.
    """
    if len(line) <= target_chars:
        return [line]

    sentences = [s for s in _SENTENCE_SPLIT_PATTERN.split(line) if s]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}" if current else sentence
        if len(candidate) > target_chars and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)

    final: list[str] = []
    for piece in pieces:
        if len(piece) <= target_chars:
            final.append(piece)
        else:
            final.extend(piece[i : i + target_chars] for i in range(0, len(piece), target_chars))
    return final
