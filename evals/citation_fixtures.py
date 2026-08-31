"""evals/citation_fixtures.py -- hand-labelled (AgentAnswer, DocumentIndex)
pairs V3 citation validity (`verify.citation.run_v3_checks`) is measured
against -- one fixture per §7.4 sub-check failure mode, mirroring
`evals/fixtures.py`'s "one fixture per failure class" convention for the
verifier as a whole.

M4's golden set never produced `kind="filing"` Evidence (nothing in the
codebase generated it before M5's RAG tools existed), so citation
precision was previously unmeasurable against anything but a vacuous,
zero-citation case. These fixtures close that gap, built through the real
`agent.document_index.LedgerDocumentIndex` from a real ledger shape --
exactly what a `retrieve_company_filings` tool call produces in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from tests.unit.verify.builders import build_answer, build_claim, build_evidence

from quantagent.agent.document_index import LedgerDocumentIndex, build_document_index
from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.ledger import Ledger, ToolCallRecord
from quantagent.contracts.provenance import Provenance
from quantagent.contracts.tools import (
    RETRIEVE_COMPANY_FILINGS,
    RetrieveCompanyFilingsOutput,
    RetrievedFilingChunk,
)

_EXCERPT = "Our supply chain is concentrated among a small number of foundries."


@dataclass(frozen=True, slots=True)
class CitationFixture:
    name: str
    answer: AgentAnswer
    document_index: LedgerDocumentIndex
    expected_verdict: str  # "PASS" | "FAIL"
    expected_check_suffix: str | None  # e.g. ":document_exists"; None for a clean pass


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


def _clean_chunk_and_index() -> tuple[RetrievedFilingChunk, LedgerDocumentIndex]:
    chunk = _chunk()
    return chunk, _document_index(chunk)


_CLEAN_CHUNK, _CLEAN_INDEX = _clean_chunk_and_index()

FIXTURE_CLEAN_FILING_CITATION = CitationFixture(
    name="clean_filing_citation",
    answer=build_answer(
        claims=[
            build_claim(
                claim_type="factual",
                evidence_ids=["ev_filing"],
                text="Supply chain is concentrated.",
            )
        ],
        evidence=[
            build_evidence(
                evidence_id="ev_filing",
                kind="filing",
                ref=_CLEAN_CHUNK.chunk_id,
                excerpt=_CLEAN_CHUNK.excerpt,
                char_span=_CLEAN_CHUNK.char_span,
                source_title="NVDA 10-K",
                source_url=_CLEAN_CHUNK.source_url,
                source_tier=_CLEAN_CHUNK.source_tier,
            )
        ],
    ),
    document_index=_CLEAN_INDEX,
    expected_verdict="PASS",
    expected_check_suffix=None,
)

FIXTURE_FABRICATED_CITATION = CitationFixture(
    name="fabricated_citation",
    answer=build_answer(
        claims=[build_claim(claim_type="factual", evidence_ids=["ev_filing"], text="x")],
        evidence=[build_evidence(evidence_id="ev_filing", kind="filing", ref="does-not-exist")],
    ),
    document_index=_CLEAN_INDEX,
    expected_verdict="FAIL",
    expected_check_suffix=":document_exists",
)

FIXTURE_EXCERPT_MISMATCH = CitationFixture(
    name="excerpt_mismatch",
    answer=build_answer(
        claims=[build_claim(claim_type="factual", evidence_ids=["ev_filing"], text="x")],
        evidence=[
            build_evidence(
                evidence_id="ev_filing",
                kind="filing",
                ref=_CLEAN_CHUNK.chunk_id,
                excerpt="This sentence was never in the filing at all.",
                char_span=_CLEAN_CHUNK.char_span,
                source_title="NVDA 10-K",
                source_url=_CLEAN_CHUNK.source_url,
                source_tier=_CLEAN_CHUNK.source_tier,
            )
        ],
    ),
    document_index=_CLEAN_INDEX,
    expected_verdict="FAIL",
    expected_check_suffix=":excerpt_match",
)

FIXTURE_WRONG_SOURCE_URL = CitationFixture(
    name="wrong_source_url",
    answer=build_answer(
        claims=[build_claim(claim_type="factual", evidence_ids=["ev_filing"], text="x")],
        evidence=[
            build_evidence(
                evidence_id="ev_filing",
                kind="filing",
                ref=_CLEAN_CHUNK.chunk_id,
                excerpt=_CLEAN_CHUNK.excerpt,
                char_span=_CLEAN_CHUNK.char_span,
                source_title="NVDA 10-K",
                source_url="https://not-the-real-filing-url.example.com",
                source_tier=_CLEAN_CHUNK.source_tier,
            )
        ],
    ),
    document_index=_CLEAN_INDEX,
    expected_verdict="FAIL",
    expected_check_suffix=":source_url",
)

_T4_CHUNK = _chunk(source_tier="T4")
_T4_INDEX = _document_index(_T4_CHUNK)

FIXTURE_T4_TIER_ON_CAUSAL_CLAIM = CitationFixture(
    name="t4_tier_on_causal_claim",
    answer=build_answer(
        claims=[build_claim(claim_type="causal", evidence_ids=["ev_filing"], text="x")],
        evidence=[
            build_evidence(
                evidence_id="ev_filing",
                kind="filing",
                ref=_T4_CHUNK.chunk_id,
                excerpt=_T4_CHUNK.excerpt,
                char_span=_T4_CHUNK.char_span,
                source_title="NVDA 10-K",
                source_url=_T4_CHUNK.source_url,
                source_tier="T4",
            )
        ],
    ),
    document_index=_T4_INDEX,
    expected_verdict="FAIL",
    expected_check_suffix=":source_tier",
)

CITATION_FIXTURES: list[CitationFixture] = [
    FIXTURE_CLEAN_FILING_CITATION,
    FIXTURE_FABRICATED_CITATION,
    FIXTURE_EXCERPT_MISMATCH,
    FIXTURE_WRONG_SOURCE_URL,
    FIXTURE_T4_TIER_ON_CAUSAL_CLAIM,
]
