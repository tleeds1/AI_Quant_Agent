"""guardrails/refusals.py -- builds a schema-valid `AgentAnswer` for a
guardrail refusal or block (architecture.md §8.3: state what cannot be
answered, plus what can).

Templates come from `rules/policy.yaml` verbatim -- no user-controlled text
is interpolated into the response, so attacker-controlled input is never
reflected back into a refusal.

`agent/loop.py::_build_refusal_answer`/`_build_safe_fallback_answer` build
the analogous shapes for OUT_OF_SCOPE-by-intent-classifier and
verification-failure paths; guardrails/ cannot import agent/ (layering),
so this is a parallel, guardrails-owned builder for the guardrail-specific
paths, explicitly distinguished in `summary`/`limitations` so an audit
reader can tell the three apart.
"""

from __future__ import annotations

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.verification import VerificationReport
from quantagent.guardrails.policy import PolicyConfig, get_default_policy
from quantagent.guardrails.types import GuardrailDecision


def build_refusal_answer(
    *, trace_id: str, decision: GuardrailDecision, policy: PolicyConfig | None = None
) -> AgentAnswer:
    """For an inbound BLOCK: the request never reaches the agent loop."""
    resolved_policy = policy or get_default_policy()
    is_prohibited_request = decision.category == "prohibited_request"
    subcategory = decision.metadata.get("matched_group") if is_prohibited_request else None
    summary = resolved_policy.refusal_template(decision.category, subcategory)
    return AgentAnswer(
        trace_id=trace_id,
        scope="GUARDRAIL_REFUSAL",
        decision="INSUFFICIENT_EVIDENCE",
        confidence=0.0,
        confidence_basis=[f"inbound_guardrail_block:{decision.category}"],
        risk_level="LOW",
        horizon="n/a",
        summary=summary,
        claims=[],
        evidence=[],
        quant_metrics={},
        constraints_checked=[],
        limitations=[f"Request blocked by inbound guardrail: {decision.reason}"],
        disclosures=[resolved_policy.disclosure_template("not_investment_advice")],
        verification=VerificationReport(verdict="PASS", checks=0, warnings=0, repair_attempts=0),
    )


def build_outbound_block_answer(
    *, trace_id: str, decision: GuardrailDecision, policy: PolicyConfig | None = None
) -> AgentAnswer:
    """For an outbound BLOCK: a fully synthesized-and-verified answer
    existed but failed a policy check on release. Safe-fallback shape,
    same rationale as `agent/loop.py::_build_safe_fallback_answer` but
    explicitly labelled as an outbound-policy block, not a verification
    failure, so the two are never confused in an audit trail.
    """
    resolved_policy = policy or get_default_policy()
    return AgentAnswer(
        trace_id=trace_id,
        scope="GUARDRAIL_BLOCK",
        decision="INSUFFICIENT_EVIDENCE",
        confidence=0.0,
        confidence_basis=[f"outbound_guardrail_block:{decision.category}"],
        risk_level="EXTREME",
        horizon="n/a",
        summary=(
            "This analysis could not be released because it failed an outbound compliance "
            "check. I can attempt a narrower version of the question instead."
        ),
        claims=[],
        evidence=[],
        quant_metrics={},
        constraints_checked=[],
        limitations=[f"Outbound guardrail blocked release: {decision.reason}"],
        disclosures=[resolved_policy.disclosure_template("not_investment_advice")],
        verification=VerificationReport(verdict="FAIL", checks=0, warnings=0, repair_attempts=0),
    )
