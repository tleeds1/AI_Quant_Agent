from __future__ import annotations

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.evidence import Claim, Evidence
from quantagent.contracts.metrics import MetricValue
from quantagent.contracts.verification import VerificationReport
from quantagent.verify.structural import run_v1_checks
from tests.unit.agent.builders import build_provenance


def _evidence(evidence_id: str, ref: str = "m1", kind: str = "metric") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        kind=kind,  # type: ignore[arg-type]
        ref=ref,
        excerpt=None,
        char_span=None,
        source_title="get_holdings",
        source_url=None,
        source_tier=None,
        published_at=None,
        retrieval_score=None,
    )


def _claim(claim_id: str, evidence_ids: list[str]) -> Claim:
    return Claim(claim_id=claim_id, text="x", claim_type="factual", evidence_ids=evidence_ids)


def _metric(metric_id: str = "m1") -> MetricValue:
    return MetricValue(
        metric_id=metric_id, value=1.0, unit="ratio", method="test", provenance=build_provenance()
    )


def _answer(
    claims: list[Claim] | None = None,
    evidence: list[Evidence] | None = None,
    *,
    quant_metrics: dict[str, MetricValue] | None = None,
    scope: str = "PORTFOLIO",
    decision: str = "HOLD",
) -> AgentAnswer:
    return AgentAnswer(
        trace_id="tr_1",
        scope=scope,
        decision=decision,  # type: ignore[arg-type]
        confidence=0.5,
        confidence_basis=[],
        risk_level="LOW",
        horizon="n/a",
        summary="s",
        claims=claims or [],
        evidence=evidence or [],
        quant_metrics=quant_metrics or {},
        constraints_checked=[],
        limitations=["none"],
        disclosures=[],
        verification=VerificationReport(verdict="PASS", checks=0, warnings=0, repair_attempts=0),
    )


# ---- evidence resolution (folded in from M3) -------------------------------


def test_no_claims_passes_trivially() -> None:
    results = run_v1_checks(_answer())
    assert not any(r.check_id == "v1.evidence_resolution" for r in results)


def test_all_claims_resolve_passes() -> None:
    results = run_v1_checks(_answer([_claim("c1", ["ev1"])], [_evidence("ev1")]))
    resolution = [r for r in results if r.check_id == "v1.evidence_resolution"]
    assert len(resolution) == 1
    assert resolution[0].verdict == "PASS"


def test_dangling_evidence_id_fails() -> None:
    results = run_v1_checks(_answer([_claim("c1", ["ev_missing"])], [_evidence("ev1")]))
    resolution = [r for r in results if r.check_id == "v1.evidence_resolution"]
    assert resolution[0].verdict == "FAIL"
    assert resolution[0].claim_id == "c1"


def test_one_dangling_among_several_still_fails() -> None:
    results = run_v1_checks(
        _answer([_claim("c1", ["ev1"]), _claim("c2", ["ev_missing"])], [_evidence("ev1")])
    )
    resolution = [r for r in results if r.check_id == "v1.evidence_resolution"]
    assert len(resolution) == 2
    assert {r.claim_id: r.verdict for r in resolution} == {"c1": "PASS", "c2": "FAIL"}


# ---- metric ref resolution (new in M4) -------------------------------------


def test_metric_kind_evidence_ref_resolves() -> None:
    results = run_v1_checks(
        _answer(evidence=[_evidence("ev1", ref="m1")], quant_metrics={"m1": _metric()})
    )
    check = [r for r in results if r.check_id == "v1.metric_ref_resolution"]
    assert check[0].verdict == "PASS"


def test_metric_kind_evidence_ref_does_not_resolve() -> None:
    results = run_v1_checks(
        _answer(evidence=[_evidence("ev1", ref="missing")], quant_metrics={"m1": _metric()})
    )
    check = [r for r in results if r.check_id == "v1.metric_ref_resolution"]
    assert check[0].verdict == "FAIL"
    assert check[0].evidence_id == "ev1"


def test_non_metric_kind_evidence_skipped() -> None:
    results = run_v1_checks(_answer(evidence=[_evidence("ev1", ref="whatever", kind="filing")]))
    assert not any(r.check_id == "v1.metric_ref_resolution" for r in results)


# ---- decision-allowed-for-scope (new in M4) --------------------------------


def test_decision_allowed_for_portfolio_scope() -> None:
    results = run_v1_checks(_answer(scope="PORTFOLIO", decision="BUY"))
    check = next(r for r in results if r.check_id == "v1.decision_scope")
    assert check.verdict == "PASS"


def test_decision_not_allowed_for_out_of_scope() -> None:
    results = run_v1_checks(_answer(scope="OUT_OF_SCOPE", decision="BUY"))
    check = next(r for r in results if r.check_id == "v1.decision_scope")
    assert check.verdict == "FAIL"


def test_out_of_scope_with_insufficient_evidence_passes() -> None:
    results = run_v1_checks(_answer(scope="OUT_OF_SCOPE", decision="INSUFFICIENT_EVIDENCE"))
    check = next(r for r in results if r.check_id == "v1.decision_scope")
    assert check.verdict == "PASS"


def test_unrecognized_scope_fails() -> None:
    results = run_v1_checks(_answer(scope="SOMETHING_NEW", decision="HOLD"))
    check = next(r for r in results if r.check_id == "v1.decision_scope")
    assert check.verdict == "FAIL"
    assert "unrecognized scope" in check.message
