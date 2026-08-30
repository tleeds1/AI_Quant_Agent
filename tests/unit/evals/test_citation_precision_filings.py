"""tests/unit/evals/test_citation_precision_filings.py -- V3 citation
validity (`verify.citation.run_v3_checks`) against `kind="filing"`
Evidence, built through the real `agent.document_index.LedgerDocumentIndex`
from a real ledger shape.

M4's golden set (`tests/unit/evals/fixtures.py`) only ever exercises
`kind="metric"` Evidence -- nothing produced `kind="filing"` Evidence
before M5's RAG tools existed. This closes that real gap: one fixture per
§7.4 sub-check failure mode, mirroring M4's "one fixture per failure
class" precedent, run directly against `run_v3_checks` (not the full
`run_verification` pipeline, which would also require mocking the V5 LLM
critic for no benefit here -- V3 is independently callable and this is a
V3-scoped test).
"""

from __future__ import annotations

from datetime import date, datetime

from quantagent.agent.document_index import LedgerDocumentIndex, build_document_index
from quantagent.contracts.ledger import Ledger, ToolCallRecord
from quantagent.contracts.provenance import Provenance
from quantagent.contracts.tools import (
    RETRIEVE_COMPANY_FILINGS,
    RetrieveCompanyFilingsOutput,
    RetrievedFilingChunk,
)
from quantagent.verify.citation import run_v3_checks
from tests.unit.verify.builders import build_answer, build_claim, build_evidence

_EXCERPT = "Our supply chain is concentrated among a small number of foundries."


def _chunk(**overrides: object) -> RetrievedFilingChunk:
    defaults: dict[str, object] = dict(
        chunk_id="acc-1#1A#0000",
        ticker="NVDA",
        cik="0001045810",
        form_type="10-K",
        filed_at=date(2024, 2, 21),
        item="1A",
        section_path="10-K#Item 1A",
        excerpt=_EXCERPT,
        char_span=(0, len(_EXCERPT)),
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/x/nvda.htm",
        source_tier="T1",
        retrieval_score=1.0,
    )
    defaults.update(overrides)
    return RetrievedFilingChunk(**defaults)  # type: ignore[arg-type]


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


def _document_index(chunk: RetrievedFilingChunk) -> LedgerDocumentIndex:
    output = RetrieveCompanyFilingsOutput(ticker="NVDA", chunks=[chunk], provenance=_provenance())
    ledger = Ledger(
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
    return build_document_index(ledger)


def test_clean_filing_citation_passes_every_subcheck() -> None:
    chunk = _chunk()
    index = _document_index(chunk)
    evidence = build_evidence(
        kind="filing",
        ref=chunk.chunk_id,
        excerpt=chunk.excerpt,
        char_span=chunk.char_span,
        source_title="NVDA 10-K",
        source_url=chunk.source_url,
        source_tier=chunk.source_tier,
    )
    answer = build_answer(
        claims=[build_claim(claim_type="factual", evidence_ids=[evidence.evidence_id])],
        evidence=[evidence],
    )

    results = run_v3_checks(answer, document_index=index)

    assert results
    assert not any(r.verdict == "FAIL" for r in results)


def test_fabricated_citation_fails_document_exists() -> None:
    index = _document_index(_chunk())
    evidence = build_evidence(kind="filing", ref="does-not-exist-in-the-index")
    answer = build_answer(
        claims=[build_claim(claim_type="factual", evidence_ids=[evidence.evidence_id])],
        evidence=[evidence],
    )

    results = run_v3_checks(answer, document_index=index)

    assert any(r.check_id.endswith(":document_exists") and r.verdict == "FAIL" for r in results)


def test_excerpt_not_matching_stored_text_fails_excerpt_match() -> None:
    chunk = _chunk()
    index = _document_index(chunk)
    evidence = build_evidence(
        kind="filing",
        ref=chunk.chunk_id,
        excerpt="This sentence was never in the filing at all.",
        char_span=chunk.char_span,
        source_url=chunk.source_url,
        source_tier=chunk.source_tier,
    )
    answer = build_answer(
        claims=[build_claim(claim_type="factual", evidence_ids=[evidence.evidence_id])],
        evidence=[evidence],
    )

    results = run_v3_checks(answer, document_index=index)

    assert any(r.check_id.endswith(":excerpt_match") and r.verdict == "FAIL" for r in results)


def test_wrong_source_url_fails_source_url_check() -> None:
    chunk = _chunk()
    index = _document_index(chunk)
    evidence = build_evidence(
        kind="filing",
        ref=chunk.chunk_id,
        excerpt=chunk.excerpt,
        char_span=chunk.char_span,
        source_url="https://not-the-real-filing-url.example.com",
        source_tier=chunk.source_tier,
    )
    answer = build_answer(
        claims=[build_claim(claim_type="factual", evidence_ids=[evidence.evidence_id])],
        evidence=[evidence],
    )

    results = run_v3_checks(answer, document_index=index)

    assert any(r.check_id.endswith(":source_url") and r.verdict == "FAIL" for r in results)


def test_t4_tier_citation_fails_a_causal_claims_tier_floor() -> None:
    chunk = _chunk(source_tier="T4")
    index = _document_index(chunk)
    evidence = build_evidence(
        kind="filing",
        ref=chunk.chunk_id,
        excerpt=chunk.excerpt,
        char_span=chunk.char_span,
        source_url=chunk.source_url,
        source_tier="T4",
    )
    answer = build_answer(
        claims=[build_claim(claim_type="causal", evidence_ids=[evidence.evidence_id])],
        evidence=[evidence],
    )

    results = run_v3_checks(answer, document_index=index)

    assert any(r.check_id.endswith(":source_tier") and r.verdict == "FAIL" for r in results)
