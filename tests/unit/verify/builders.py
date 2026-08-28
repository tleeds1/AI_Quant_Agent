"""tests/unit/verify/builders.py -- shared test builders for verify/ tests."""

from __future__ import annotations

from datetime import datetime

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.evidence import Claim, Evidence, SourceTier
from quantagent.contracts.ledger import Ledger, ToolCallRecord, ToolCallStatus
from quantagent.contracts.verification import VerificationReport
from quantagent.verify.citation import DocumentMetadata


class FakeDocumentIndex:
    """In-memory `DocumentIndex`. A document not `seed()`-ed is simply
    absent (fabricated-citation behavior).
    """

    def __init__(self) -> None:
        self._docs: dict[str, tuple[DocumentMetadata, str]] = {}
        self._urls: dict[str, str] = {}

    def seed(
        self,
        document_id: str,
        *,
        chunk_text: str,
        source_tier: SourceTier,
        published_at: datetime,
        source_url: str | None = None,
    ) -> None:
        self._docs[document_id] = (
            DocumentMetadata(
                document_id=document_id, source_tier=source_tier, published_at=published_at
            ),
            chunk_text,
        )
        if source_url is not None:
            self._urls[document_id] = source_url

    def get_metadata(self, document_id: str) -> DocumentMetadata | None:
        entry = self._docs.get(document_id)
        return entry[0] if entry else None

    def get_chunk_text(self, document_id: str) -> str | None:
        entry = self._docs.get(document_id)
        return entry[1] if entry else None

    def resolves_source_url(self, document_id: str, source_url: str) -> bool:
        return self._urls.get(document_id) == source_url


class RaisingDocumentIndex:
    """Every method raises -- used to prove metric/transaction/market_data
    evidence never calls into the index at all.
    """

    def get_metadata(self, document_id: str) -> DocumentMetadata | None:
        raise AssertionError("DocumentIndex should not be called for this evidence kind")

    def get_chunk_text(self, document_id: str) -> str | None:
        raise AssertionError("DocumentIndex should not be called for this evidence kind")

    def resolves_source_url(self, document_id: str, source_url: str) -> bool:
        raise AssertionError("DocumentIndex should not be called for this evidence kind")


def build_evidence(evidence_id: str = "ev1", **overrides: object) -> Evidence:
    defaults: dict[str, object] = dict(
        evidence_id=evidence_id,
        kind="filing",
        ref="doc1",
        excerpt=None,
        char_span=None,
        source_title="10-K",
        source_url=None,
        source_tier=None,
        published_at=None,
        retrieval_score=None,
    )
    defaults.update(overrides)
    return Evidence(**defaults)  # type: ignore[arg-type]


def build_claim(
    claim_id: str = "c1", evidence_ids: list[str] | None = None, **overrides: object
) -> Claim:
    defaults: dict[str, object] = dict(
        claim_id=claim_id, text="x", claim_type="factual", evidence_ids=evidence_ids or ["ev1"]
    )
    defaults.update(overrides)
    return Claim(**defaults)  # type: ignore[arg-type]


def build_answer(**overrides: object) -> AgentAnswer:
    defaults: dict[str, object] = dict(
        trace_id="tr_1",
        scope="PORTFOLIO",
        decision="HOLD",
        confidence=0.5,
        confidence_basis=[],
        risk_level="LOW",
        horizon="n/a",
        summary="s",
        claims=[],
        evidence=[],
        quant_metrics={},
        constraints_checked=[],
        limitations=["none"],
        disclosures=[],
        verification=VerificationReport(verdict="PASS", checks=0, warnings=0, repair_attempts=0),
    )
    defaults.update(overrides)
    return AgentAnswer(**defaults)  # type: ignore[arg-type]


def build_tool_call_record(
    call_id: str = "tc1", status: ToolCallStatus = "OK", **overrides: object
) -> ToolCallRecord:
    defaults: dict[str, object] = dict(
        call_id=call_id,
        tool_name="calculate_portfolio_var",
        args={},
        args_hash="h",
        status=status,
        latency_ms=10,
        cost_usd=0.0,
        result=None,
        error=None,
    )
    defaults.update(overrides)
    return ToolCallRecord(**defaults)  # type: ignore[arg-type]


def build_ledger(calls: list[ToolCallRecord] | None = None, **overrides: object) -> Ledger:
    defaults: dict[str, object] = dict(trace_id="tr_1", calls=calls or [], numeric_index={})
    defaults.update(overrides)
    return Ledger(**defaults)  # type: ignore[arg-type]
