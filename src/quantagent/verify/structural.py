"""verify/structural.py -- V1: schema & contract layer (architecture.md
§7.1).

Pydantic strict validation is automatic (every contracts/ model already is
strict) and needs no code here. `limitations` non-empty is enforced by
`AgentAnswer.limitations: list[str] = Field(min_length=1)` itself -- a
schema-valid `AgentAnswer` cannot violate it, so it needs no runtime check
either. That leaves three genuinely runtime checks:

  1. every Claim.evidence_ids resolves to an Evidence.evidence_id present
     in the same answer (M3's original check, folded in here)
  2. every Evidence.ref of kind "metric" resolves to a quant_metrics key
  3. decision is in the allowed set for the given scope
"""

from __future__ import annotations

from quantagent.contracts.answer import AgentAnswer, Decision
from quantagent.verify.types import CheckResult

# `AgentAnswer.scope` is an untyped `str`, not a Literal (contracts/answer.py
# is frozen -- flagged, not fixed here). The two real values produced by
# agent/loop.py today are "PORTFOLIO" (normal analysis and the safe-fallback
# path) and "OUT_OF_SCOPE" (the REFUSE path, decision always
# INSUFFICIENT_EVIDENCE there).
#
# An unrecognized scope string is treated as a contract violation -> FAIL,
# never silently passed (I8: no silent degradation). A future scope value
# MUST be added here explicitly; forgetting to do so fails loudly instead of
# quietly accepting an unvetted (scope, decision) combination.
_ALLOWED_DECISIONS_BY_SCOPE: dict[str, set[Decision]] = {
    "PORTFOLIO": {
        "BUY",
        "HOLD",
        "SELL",
        "REDUCE",
        "HEDGE",
        "NO_ACTION",
        "INSUFFICIENT_EVIDENCE",
    },
    "OUT_OF_SCOPE": {"INSUFFICIENT_EVIDENCE"},
}


def run_v1_checks(answer: AgentAnswer) -> list[CheckResult]:
    """Entrypoint for the verify/verdict.py orchestrator."""
    results: list[CheckResult] = []
    results.extend(_check_evidence_resolution(answer))
    results.extend(_check_metric_ref_resolution(answer))
    results.append(_check_decision_allowed_for_scope(answer))
    return results


def _check_evidence_resolution(answer: AgentAnswer) -> list[CheckResult]:
    """One CheckResult per Claim."""
    resolvable_ids = {evidence.evidence_id for evidence in answer.evidence}
    results: list[CheckResult] = []
    for claim in answer.claims:
        dangling = [eid for eid in claim.evidence_ids if eid not in resolvable_ids]
        if dangling:
            results.append(
                CheckResult(
                    layer="V1",
                    check_id="v1.evidence_resolution",
                    verdict="FAIL",
                    message=(
                        f"claim {claim.claim_id} references unresolved evidence_ids: {dangling}"
                    ),
                    claim_id=claim.claim_id,
                )
            )
        else:
            results.append(
                CheckResult(
                    layer="V1",
                    check_id="v1.evidence_resolution",
                    verdict="PASS",
                    message=f"claim {claim.claim_id}: all evidence_ids resolve",
                    claim_id=claim.claim_id,
                )
            )
    return results


def _check_metric_ref_resolution(answer: AgentAnswer) -> list[CheckResult]:
    """Per prompts/synthesis/answer.v1.jinja's own output-contract
    instructions, `Evidence.ref` for `kind="metric"` is set to the metric's
    `metric_id`; this checks `ref in answer.quant_metrics`.
    """
    results: list[CheckResult] = []
    for evidence in answer.evidence:
        if evidence.kind != "metric":
            continue
        if evidence.ref in answer.quant_metrics:
            results.append(
                CheckResult(
                    layer="V1",
                    check_id="v1.metric_ref_resolution",
                    verdict="PASS",
                    message=f"evidence {evidence.evidence_id}: ref '{evidence.ref}' resolves",
                    evidence_id=evidence.evidence_id,
                )
            )
        else:
            results.append(
                CheckResult(
                    layer="V1",
                    check_id="v1.metric_ref_resolution",
                    verdict="FAIL",
                    message=(
                        f"evidence {evidence.evidence_id} has kind='metric' but ref "
                        f"'{evidence.ref}' is not a key in quant_metrics "
                        f"({sorted(answer.quant_metrics)})"
                    ),
                    evidence_id=evidence.evidence_id,
                )
            )
    return results


def _check_decision_allowed_for_scope(answer: AgentAnswer) -> CheckResult:
    allowed = _ALLOWED_DECISIONS_BY_SCOPE.get(answer.scope)
    if allowed is None:
        return CheckResult(
            layer="V1",
            check_id="v1.decision_scope",
            verdict="FAIL",
            message=(
                f"unrecognized scope '{answer.scope}': no allowed-decision table entry "
                "exists for it; treated as a contract violation rather than passed"
            ),
        )
    if answer.decision in allowed:
        return CheckResult(
            layer="V1",
            check_id="v1.decision_scope",
            verdict="PASS",
            message=f"decision '{answer.decision}' is allowed for scope '{answer.scope}'",
        )
    return CheckResult(
        layer="V1",
        check_id="v1.decision_scope",
        verdict="FAIL",
        message=(
            f"decision '{answer.decision}' is not allowed for scope '{answer.scope}' "
            f"(allowed: {sorted(allowed)})"
        ),
    )
