"""agent/document_index.py -- builds a `verify.citation.DocumentIndex` from
the `Ledger` itself, not by re-querying the repository (docs/PROGRESS.md's
resolved M5 design).

Two constraints motivate this: `verify.citation.DocumentIndex`'s methods
are synchronous (verify/ is pure/sync by design), but a live chunk
repository is necessarily async; and re-querying the DB at verify time
would risk a second, potentially-inconsistent round trip, breaking
architecture.md §9.2's bit-for-bit trace-replay guarantee (a live index can
change between synthesis and a nightly replay; a frozen ledger snapshot
cannot).

Nice side effect: a chunk the injection classifier quarantined at
tool-call time (`tools/research.py`) never entered the ledger's chunk list
in the first place, so this composes correctly with retrieval-time
screening with no extra coordination -- a quarantined chunk can never be
accidentally certified as valid evidence here.

Structurally satisfies `verify.citation.DocumentIndex` via duck typing
(`verify/` cannot import `rag/` or `agent/`, so it never imports this
class -- dependency inversion, same as M4).
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any

from pydantic import BaseModel

from quantagent.contracts.ledger import Ledger
from quantagent.contracts.tools import (
    RETRIEVE_COMPANY_FILINGS,
    RETRIEVE_FILING_SECTION,
    RetrieveCompanyFilingsOutput,
    RetrievedFilingChunk,
    RetrieveFilingSectionOutput,
)
from quantagent.verify.citation import DocumentMetadata

_RAG_TOOL_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    RETRIEVE_COMPANY_FILINGS: RetrieveCompanyFilingsOutput,
    RETRIEVE_FILING_SECTION: RetrieveFilingSectionOutput,
}


def _extract_chunks(
    output_model: type[BaseModel], result: dict[str, Any]
) -> list[RetrievedFilingChunk]:
    output = output_model.model_validate(result)
    if isinstance(output, RetrieveCompanyFilingsOutput | RetrieveFilingSectionOutput):
        return output.chunks
    return []


class LedgerDocumentIndex:
    """Zero I/O, built once per request from the already-executed `Ledger`.

    `get_chunk_text` returns the chunk's `excerpt` -- the only chunk text
    that ever reached the synthesiser (see `RetrievedFilingChunk`'s
    docstring) -- not the full underlying document. V3's excerpt/char_span
    check is still meaningful: the synthesiser's `char_span` is instructed
    to describe an offset into that same excerpt, so this remains a real
    verbatim check, just scoped to what the model could actually see.
    """

    def __init__(self, ledger: Ledger) -> None:
        self._metadata: dict[str, DocumentMetadata] = {}
        self._chunk_text: dict[str, str] = {}
        self._source_url: dict[str, str] = {}
        for call in ledger.calls:
            output_model = _RAG_TOOL_OUTPUT_MODELS.get(call.tool_name)
            if output_model is None or call.result is None:
                continue
            chunks = _extract_chunks(output_model, call.result)
            for chunk in chunks:
                self._index_chunk(chunk)

    def _index_chunk(self, chunk: RetrievedFilingChunk) -> None:
        self._metadata[chunk.chunk_id] = DocumentMetadata(
            document_id=chunk.chunk_id,
            source_tier=chunk.source_tier,
            published_at=datetime.combine(chunk.filed_at, time.min),
        )
        self._chunk_text[chunk.chunk_id] = chunk.excerpt
        self._source_url[chunk.chunk_id] = chunk.source_url

    def get_metadata(self, document_id: str) -> DocumentMetadata | None:
        return self._metadata.get(document_id)

    def get_chunk_text(self, document_id: str) -> str | None:
        return self._chunk_text.get(document_id)

    def resolves_source_url(self, document_id: str, source_url: str) -> bool:
        return self._source_url.get(document_id) == source_url


def build_document_index(ledger: Ledger) -> LedgerDocumentIndex:
    return LedgerDocumentIndex(ledger)
