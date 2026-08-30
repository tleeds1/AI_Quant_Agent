"""guardrails/outbound.py -- OUTBOUND guardrails (architecture.md §8.2), run
on the fully-verified `AgentAnswer` right before release.

Two of architecture's six outbound checks are already implemented
elsewhere and out of scope here: "output schema conformance" is V1
(`verify/structural.py`); "verifier verdict gate" is `agent/loop.py`'s
existing `verification.verdict == "FAIL"` -> safe-fallback branch. This
module covers the remaining four blocking checks plus disclosure assembly
(not a check -- always runs on an ALLOWed answer).

`check_advice_framing` and the rest BLOCK to safe-fallback rather than
attempting to rewrite the answer: outbound guardrails run after the
verifier/repair pipeline has already certified this exact text (numeric
grounding, citations); silently editing it here could invalidate those
guarantees, and there is no repair budget left at this stage.
"""

from __future__ import annotations

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.errors import PolicyViolationError
from quantagent.guardrails.fail_closed import run_check_safely
from quantagent.guardrails.normalize import normalize_for_matching
from quantagent.guardrails.pii import redact_pii
from quantagent.guardrails.policy import PolicyConfig, get_default_policy
from quantagent.guardrails.types import GuardrailContext, GuardrailDecision, OutboundPayload

_DISCLOSURE_KEYS = (
    "not_investment_advice",
    "data_as_of",
    "model_limitations",
    "no_conflicts",
)


def _text_fields(answer: AgentAnswer) -> list[str]:
    return [answer.summary, *(claim.text for claim in answer.claims)]


def check_prohibited_language(
    payload: OutboundPayload, context: GuardrailContext, *, policy: PolicyConfig | None = None
) -> GuardrailDecision:
    resolved_policy = policy or get_default_policy()
    for field_text in _text_fields(payload.answer):
        normalized = normalize_for_matching(field_text)
        for pattern in resolved_policy.prohibited_language_patterns():
            if pattern.search(normalized):
                return GuardrailDecision(
                    check_id="outbound.prohibited_language",
                    action="BLOCK",
                    category="prohibited_language",
                    reason=f"matched prohibited-language pattern {pattern.pattern!r}.",
                )
    return GuardrailDecision(
        check_id="outbound.prohibited_language", action="ALLOW", category="none"
    )


def check_advice_framing(
    payload: OutboundPayload, context: GuardrailContext, *, policy: PolicyConfig | None = None
) -> GuardrailDecision:
    if context.licensed_for_suitability_advice:
        return GuardrailDecision(
            check_id="outbound.advice_framing",
            action="ALLOW",
            category="none",
            reason="tenant is licensed and configured for suitability advice.",
        )
    resolved_policy = policy or get_default_policy()
    for field_text in _text_fields(payload.answer):
        normalized = normalize_for_matching(field_text)
        for pattern in resolved_policy.advice_framing_patterns():
            if pattern.search(normalized):
                return GuardrailDecision(
                    check_id="outbound.advice_framing",
                    action="BLOCK",
                    category="advice_framing",
                    reason=f"matched advice-framing pattern {pattern.pattern!r}.",
                )
    return GuardrailDecision(check_id="outbound.advice_framing", action="ALLOW", category="none")


def check_pii_egress(
    payload: OutboundPayload, context: GuardrailContext, *, policy: PolicyConfig | None = None
) -> GuardrailDecision:
    excerpts = [e.excerpt for e in payload.answer.evidence if e.excerpt]
    matched: list[str] = []
    for field_text in [*_text_fields(payload.answer), *excerpts]:
        matched.extend(redact_pii(field_text, policy=policy).matched_pattern_ids)
    if not matched:
        return GuardrailDecision(check_id="outbound.pii_egress", action="ALLOW", category="none")
    return GuardrailDecision(
        check_id="outbound.pii_egress",
        action="BLOCK",
        category="pii_egress",
        reason=f"PII patterns found in outbound text: {sorted(set(matched))}.",
    )


def check_leakage(
    payload: OutboundPayload, context: GuardrailContext, *, policy: PolicyConfig | None = None
) -> GuardrailDecision:
    resolved_policy = policy or get_default_policy()
    excerpts = [e.excerpt for e in payload.answer.evidence if e.excerpt]
    for field_text in [*_text_fields(payload.answer), *excerpts]:
        for pattern in resolved_policy.leakage_patterns():
            if pattern.search(field_text):
                return GuardrailDecision(
                    check_id="outbound.leakage",
                    action="BLOCK",
                    category="leakage",
                    reason="matched a key/prompt-leakage pattern in outbound text.",
                )
    return GuardrailDecision(check_id="outbound.leakage", action="ALLOW", category="none")


_OUTBOUND_CHECKS: tuple[object, ...] = (
    check_prohibited_language,
    check_advice_framing,
    check_pii_egress,
    check_leakage,
)


def run_outbound_checks(
    payload: OutboundPayload,
    context: GuardrailContext,
    *,
    checks: tuple[object, ...] | None = None,
) -> GuardrailDecision:
    for check in checks or _OUTBOUND_CHECKS:
        decision = run_check_safely(check, payload, context)  # type: ignore[arg-type]
        if decision.action == "BLOCK":
            return decision
    return GuardrailDecision(check_id="outbound.all", action="ALLOW", category="none")


def enforce_outbound(payload: OutboundPayload, context: GuardrailContext) -> None:
    decision = run_outbound_checks(payload, context)
    if decision.action == "BLOCK":
        raise PolicyViolationError(decision.reason)


def apply_disclosures(answer: AgentAnswer, *, policy: PolicyConfig | None = None) -> AgentAnswer:
    """Always runs on an ALLOWed answer -- not a check. REPLACES
    `answer.disclosures` with the mandatory set from `rules/policy.yaml`,
    rather than appending to whatever the synthesiser wrote: today the LLM
    still free-composes this field, which is exactly what guideline.md §9
    forbids ("Disclosures are appended by code from a template, never
    generated by the LLM") -- this is the guardrail that makes that true at
    release time. Idempotent (running twice yields the same list).
    `simulated_scenario` is only added when `summary`/a `Claim.text` marks
    something hypothetical -- architecture.md §4.6's convention is the
    literal word "hypothetical". Never touches `summary`/`claims`.
    """
    resolved_policy = policy or get_default_policy()
    disclosures = [resolved_policy.disclosure_template(key) for key in _DISCLOSURE_KEYS]
    if _mentions_hypothetical_scenario(answer):
        disclosures.append(resolved_policy.disclosure_template("simulated_scenario"))
    return answer.model_copy(update={"disclosures": disclosures})


def _mentions_hypothetical_scenario(answer: AgentAnswer) -> bool:
    haystacks = [answer.summary, *(claim.text for claim in answer.claims)]
    return any("hypothetical" in text.lower() for text in haystacks)
