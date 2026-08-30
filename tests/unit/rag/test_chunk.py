"""tests/unit/rag/test_chunk.py"""

from __future__ import annotations

from datetime import date

from quantagent.rag.chunk import CHUNK_TARGET_CHARS, FilingIdentity, chunk_filing

_IDENTITY = FilingIdentity(
    ticker="NVDA",
    cik="0001045810",
    accession_no="0001045810-24-000123",
    form_type="10-K",
    filed_at=date(2024, 2, 21),
    period_of_report=date(2024, 1, 28),
)


def _wrap(*paragraphs: str) -> str:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f"<html><body>{body}</body></html>"


def test_toc_entries_are_not_mistaken_for_the_real_heading() -> None:
    html = _wrap(
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 1. Business",
        "Our real business description.",
        "Item 1A. Risk Factors",
        "Our real risk factors.",
    )
    chunks = chunk_filing(html, _IDENTITY)
    business_text = "\n".join(c.text for c in chunks if c.metadata.item == "1")
    risk_text = "\n".join(c.text for c in chunks if c.metadata.item == "1A")
    assert "real business description" in business_text
    assert "real risk factors" in risk_text


def test_never_splits_across_an_item_boundary() -> None:
    html = _wrap(
        "Item 1A. Risk Factors",
        "Risk paragraph.",
        "Item 7. Management Discussion and Analysis",
        "MD&A paragraph.",
    )
    chunks = chunk_filing(html, _IDENTITY)
    for chunk in chunks:
        assert "MD&A paragraph" not in chunk.text or chunk.metadata.item == "7"
        assert "Risk paragraph" not in chunk.text or chunk.metadata.item == "1A"


def test_no_recognisable_item_heading_degrades_to_unknown_section() -> None:
    html = _wrap("Just some prose with no item headings at all.")
    chunks = chunk_filing(html, _IDENTITY)
    assert len(chunks) == 1
    assert chunks[0].metadata.item == "UNKNOWN"


def test_long_section_splits_into_multiple_chunks_near_target_size() -> None:
    long_paragraph = "Risk detail sentence. " * 400  # comfortably over CHUNK_TARGET_CHARS
    html = _wrap("Item 1A. Risk Factors", long_paragraph)
    chunks = chunk_filing(html, _IDENTITY)
    assert len(chunks) > 1
    assert all(c.metadata.item == "1A" for c in chunks)
    # Soft target: each chunk may run a little over, never wildly over.
    assert all(len(c.text) < CHUNK_TARGET_CHARS * 1.5 for c in chunks)


def test_chunk_id_is_stable_and_unique_per_chunk() -> None:
    long_paragraph = "Risk detail sentence. " * 400
    html = _wrap("Item 1A. Risk Factors", long_paragraph)
    chunks = chunk_filing(html, _IDENTITY)
    chunk_ids = [c.metadata.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(cid.startswith(f"{_IDENTITY.accession_no}#1A#") for cid in chunk_ids)


def test_chunk_metadata_carries_filing_identity() -> None:
    html = _wrap("Item 1. Business", "Some business text.")
    chunks = chunk_filing(html, _IDENTITY)
    metadata = chunks[0].metadata
    assert metadata.ticker == "NVDA"
    assert metadata.cik == "0001045810"
    assert metadata.form_type == "10-K"
    assert metadata.filed_at == date(2024, 2, 21)
    assert metadata.period == date(2024, 1, 28)
    assert metadata.section_path == "10-K#Item 1"


def test_a_single_line_longer_than_target_is_still_split() -> None:
    """A whole paragraph of prose in one HTML text node (no internal <br>)
    -- what BeautifulSoup.get_text emits as a single unbroken line -- must
    still be subdivided, not silently pass through as one oversized chunk.
    """
    one_giant_line = "Risk detail sentence " * 500 + "with no punctuation at all"
    html = _wrap("Item 1A. Risk Factors", one_giant_line)
    chunks = chunk_filing(html, _IDENTITY)
    assert len(chunks) > 1
    assert all(len(c.text) < CHUNK_TARGET_CHARS * 1.5 for c in chunks)
