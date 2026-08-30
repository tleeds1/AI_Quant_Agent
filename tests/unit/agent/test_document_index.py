"""tests/unit/agent/test_document_index.py"""

from __future__ import annotations

from datetime import date, datetime

from quantagent.agent.document_index import build_document_index
from quantagent.contracts.ledger import Ledger, ToolCallRecord
from quantagent.contracts.provenance import Provenance
from quantagent.contracts.tools import (
    RETRIEVE_COMPANY_FILINGS,
    RetrieveCompanyFilingsOutput,
    RetrievedFilingChunk,
)

_CHUNK = RetrievedFilingChunk(
    chunk_id="acc-1#1A#0000",
    ticker="NVDA",
    cik="0001045810",
    form_type="10-K",
    filed_at=date(2024, 2, 21),
    item="1A",
    section_path="10-K#Item 1A",
    excerpt="Our supply chain is concentrated among a small number of foundries.",
    char_span=(0, 68),
    source_url="https://www.sec.gov/Archives/edgar/data/1045810/x/nvda.htm",
    source_tier="T1",
    retrieval_score=1.0,
)


def _provenance() -> Provenance:
    return Provenance(
        tool_call_id="tc_1",
        tool_name=RETRIEVE_COMPANY_FILINGS,
        as_of=date(2024, 2, 21),
        computed_at=datetime(2024, 2, 21, 12, 0, 0),
        inputs_hash="h1",
        data_sources=["edgar"],
        estimator=None,
        sample_size=None,
        seed=None,
        warnings=[],
    )


def _ledger_with_chunk(chunk: RetrievedFilingChunk = _CHUNK) -> Ledger:
    output = RetrieveCompanyFilingsOutput(ticker="NVDA", chunks=[chunk], provenance=_provenance())
    return Ledger(
        trace_id="tr_1",
        calls=[
            ToolCallRecord(
                call_id="tc_1",
                tool_name=RETRIEVE_COMPANY_FILINGS,
                args={},
                args_hash="h",
                status="OK",
                latency_ms=10,
                cost_usd=0.0,
                result=output.model_dump(mode="json"),
                error=None,
            )
        ],
        numeric_index={},
    )


def test_indexes_a_chunk_from_a_retrieve_company_filings_call() -> None:
    index = build_document_index(_ledger_with_chunk())

    metadata = index.get_metadata("acc-1#1A#0000")
    assert metadata is not None
    assert metadata.source_tier == "T1"
    assert metadata.published_at.date() == date(2024, 2, 21)


def test_get_chunk_text_returns_the_excerpt() -> None:
    index = build_document_index(_ledger_with_chunk())

    assert index.get_chunk_text("acc-1#1A#0000") == _CHUNK.excerpt


def test_resolves_source_url_matches_only_the_real_url() -> None:
    index = build_document_index(_ledger_with_chunk())

    assert index.resolves_source_url("acc-1#1A#0000", _CHUNK.source_url)
    assert not index.resolves_source_url("acc-1#1A#0000", "https://not-the-real-url.example.com")


def test_unknown_document_id_returns_none() -> None:
    index = build_document_index(_ledger_with_chunk())

    assert index.get_metadata("does-not-exist") is None
    assert index.get_chunk_text("does-not-exist") is None
    assert not index.resolves_source_url("does-not-exist", _CHUNK.source_url)


def test_non_rag_tool_calls_are_ignored() -> None:
    ledger = Ledger(
        trace_id="tr_1",
        calls=[
            ToolCallRecord(
                call_id="tc_1",
                tool_name="calculate_portfolio_var",
                args={},
                args_hash="h",
                status="OK",
                latency_ms=10,
                cost_usd=0.0,
                result={"value": 0.02},
                error=None,
            )
        ],
        numeric_index={},
    )

    index = build_document_index(ledger)

    assert index.get_metadata("acc-1#1A#0000") is None


def test_a_rag_call_with_no_result_is_skipped_not_erroring() -> None:
    ledger = Ledger(
        trace_id="tr_1",
        calls=[
            ToolCallRecord(
                call_id="tc_1",
                tool_name=RETRIEVE_COMPANY_FILINGS,
                args={},
                args_hash="h",
                status="ERROR",
                latency_ms=10,
                cost_usd=0.0,
                result=None,
                error="boom",
            )
        ],
        numeric_index={},
    )

    index = build_document_index(ledger)

    assert index.get_metadata("acc-1#1A#0000") is None
