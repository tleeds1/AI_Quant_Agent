"""guardrails/inbound.py -- INBOUND guardrails (architecture.md §8.1),
run before any LLM call sees the question (P5 / I5).

Order matters: PII must be redacted before anything else -- including the
scope/injection/prohibited-request checks below it -- touches the text,
since those checks only need a boolean match and normalized-text search
never depends on PII being present. Rate limit and jurisdiction run first
because they're the cheapest (no text scan at all).

`check_scope` here is a narrow, cheap keyword pre-filter for egregiously
non-financial input (e.g. "write me a poem") -- it is NOT a replacement for
`agent/intent.py::classify_intent`'s LLM-based OUT_OF_SCOPE judgment on
genuinely borderline financial-adjacent questions. Its only job is to save
the cost of that LLM call on the obvious cases.
"""

from __future__ import annotations

from quantagent.contracts.errors import (
    GuardrailError,
    InjectionDetectedError,
    OutOfScopeError,
    ProhibitedRequestError,
)
from quantagent.guardrails.fail_closed import run_check_safely
from quantagent.guardrails.injection import classify_injection
from quantagent.guardrails.normalize import normalize_for_matching
from quantagent.guardrails.pii import redact_pii
from quantagent.guardrails.policy import PolicyConfig, get_default_policy
from quantagent.guardrails.types import GuardrailContext, GuardrailDecision, InboundPayload

_NON_FINANCIAL_KEYWORDS = (
    "write me a poem",
    "write a poem",
    "tell me a joke",
    "translate this",
    "write a story",
    "write code for",
    "generate an image",
)


def check_rate_limit(payload: InboundPayload, context: GuardrailContext) -> GuardrailDecision:
    if context.requests_in_window >= context.rate_limit_per_window:
        return GuardrailDecision(
            check_id="inbound.rate_limit",
            action="BLOCK",
            category="rate_limit_exceeded",
            reason=(
                f"{context.requests_in_window} requests in window >= "
                f"limit {context.rate_limit_per_window}."
            ),
        )
    return GuardrailDecision(check_id="inbound.rate_limit", action="ALLOW", category="none")


def check_jurisdiction(payload: InboundPayload, context: GuardrailContext) -> GuardrailDecision:
    allowed = context.tenant_allowed_jurisdictions
    if allowed is not None and context.jurisdiction not in allowed:
        return GuardrailDecision(
            check_id="inbound.jurisdiction",
            action="BLOCK",
            category="jurisdiction_blocked",
            reason=f"jurisdiction {context.jurisdiction!r} not in {sorted(allowed)}.",
        )
    return GuardrailDecision(check_id="inbound.jurisdiction", action="ALLOW", category="none")


def check_pii(
    payload: InboundPayload, context: GuardrailContext, *, policy: PolicyConfig | None = None
) -> GuardrailDecision:
    result = redact_pii(payload.text, policy=policy)
    if not result.matched_pattern_ids:
        return GuardrailDecision(check_id="inbound.pii", action="ALLOW", category="none")
    return GuardrailDecision(
        check_id="inbound.pii",
        action="ALLOW",
        category="pii_detected",
        reason=f"redacted PII patterns: {result.matched_pattern_ids}.",
        redacted_text=result.redacted_text,
        metadata={"matched_pattern_ids": result.matched_pattern_ids},
    )


def check_injection(
    payload: InboundPayload, context: GuardrailContext, *, policy: PolicyConfig | None = None
) -> GuardrailDecision:
    verdict = classify_injection(payload.text, policy=policy)
    if not verdict.is_injection:
        return GuardrailDecision(check_id="inbound.injection", action="ALLOW", category="none")
    return GuardrailDecision(
        check_id="inbound.injection",
        action="BLOCK",
        category="injection_detected",
        reason=f"matched injection pattern groups: {verdict.matched_group_ids}.",
        metadata={"confidence": verdict.confidence, "matched_group_ids": verdict.matched_group_ids},
    )


def check_prohibited_request(
    payload: InboundPayload, context: GuardrailContext, *, policy: PolicyConfig | None = None
) -> GuardrailDecision:
    resolved_policy = policy or get_default_policy()
    normalized = normalize_for_matching(payload.text)
    for group in resolved_policy.prohibited_request_groups():
        if any(pattern.search(normalized) for pattern in group.compiled):
            return GuardrailDecision(
                check_id="inbound.prohibited_request",
                action="BLOCK",
                category="prohibited_request",
                reason=f"matched prohibited-request group {group.group_id!r}.",
                metadata={"matched_group": group.group_id},
            )
    return GuardrailDecision(check_id="inbound.prohibited_request", action="ALLOW", category="none")


def check_scope(payload: InboundPayload, context: GuardrailContext) -> GuardrailDecision:
    normalized = normalize_for_matching(payload.text)
    if any(keyword in normalized for keyword in _NON_FINANCIAL_KEYWORDS):
        return GuardrailDecision(
            check_id="inbound.scope",
            action="BLOCK",
            category="out_of_scope",
            reason="matched a non-financial request keyword.",
        )
    return GuardrailDecision(check_id="inbound.scope", action="ALLOW", category="none")


_INBOUND_CHECKS: tuple[object, ...] = (
    check_rate_limit,
    check_jurisdiction,
    check_pii,
    check_injection,
    check_prohibited_request,
    check_scope,
)

_ERROR_BY_CATEGORY: dict[str, type[GuardrailError]] = {
    "out_of_scope": OutOfScopeError,
    "prohibited_request": ProhibitedRequestError,
    "injection_detected": InjectionDetectedError,
}


def run_inbound_checks(
    payload: InboundPayload,
    context: GuardrailContext,
    *,
    checks: tuple[object, ...] | None = None,
) -> tuple[GuardrailDecision, InboundPayload]:
    """Runs each check via `run_check_safely`, in order, folding any
    `redacted_text` forward so every later check -- and, once this returns,
    `classify_intent`/the rest of the loop -- sees the redacted text, never
    the original (I5: PII must be gone before anything else touches it).
    Short-circuits on the first BLOCK. Returns
    `(final_decision, payload_to_use_downstream)`; callers must use the
    returned payload's `.text`, not their original question.
    """
    current_payload = payload
    for check in checks or _INBOUND_CHECKS:
        decision = run_check_safely(check, current_payload, context)  # type: ignore[arg-type]
        if decision.redacted_text is not None:
            current_payload = InboundPayload(
                text=decision.redacted_text, portfolio_id=current_payload.portfolio_id
            )
        if decision.action == "BLOCK":
            return decision, current_payload
    return (
        GuardrailDecision(check_id="inbound.all", action="ALLOW", category="none"),
        current_payload,
    )


def enforce_inbound(payload: InboundPayload, context: GuardrailContext) -> InboundPayload:
    """Exception-based wrapper: raises the typed error for the three
    categories that have one, `GuardrailError` for rate-limit/jurisdiction/
    internal_error (no dedicated leaf exists for those), and returns the
    redacted payload on ALLOW.
    """
    decision, resolved_payload = run_inbound_checks(payload, context)
    if decision.action == "ALLOW":
        return resolved_payload
    error_cls = _ERROR_BY_CATEGORY.get(decision.category, GuardrailError)
    raise error_cls(decision.reason)
