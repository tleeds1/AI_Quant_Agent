"""tests/unit/guardrails/test_refusals.py"""

from __future__ import annotations

import pytest

from quantagent.guardrails.refusals import build_outbound_block_answer, build_refusal_answer
from quantagent.guardrails.types import GuardrailDecision

_INBOUND_CATEGORIES = [
    ("inbound.scope", "out_of_scope", None),
    ("inbound.injection", "injection_detected", None),
    ("inbound.prohibited_request", "prohibited_request", "insider_information"),
    ("inbound.prohibited_request", "prohibited_request", "market_manipulation"),
    ("inbound.prohibited_request", "prohibited_request", "guaranteed_returns_request"),
    ("inbound.rate_limit", "rate_limit_exceeded", None),
    ("inbound.jurisdiction", "jurisdiction_blocked", None),
]


@pytest.mark.parametrize(("check_id", "category", "matched_group"), _INBOUND_CATEGORIES)
def test_build_refusal_answer_produces_schema_valid_answer_with_both_clauses(
    check_id: str, category: str, matched_group: str | None
) -> None:
    decision = GuardrailDecision(
        check_id=check_id,
        action="BLOCK",
        category=category,  # type: ignore[arg-type]
        reason="test",
        metadata={"matched_group": matched_group} if matched_group else {},
    )
    answer = build_refusal_answer(trace_id="tr_1", decision=decision)
    assert answer.decision == "INSUFFICIENT_EVIDENCE"
    assert answer.limitations
    assert answer.disclosures
    summary_lower = answer.summary.lower()
    assert "can't" in summary_lower or "cannot" in summary_lower
    assert "instead" in summary_lower


def test_build_outbound_block_answer_is_distinguishable_from_refusal() -> None:
    decision = GuardrailDecision(
        check_id="outbound.prohibited_language",
        action="BLOCK",
        category="prohibited_language",
        reason="matched guarantee",
    )
    answer = build_outbound_block_answer(trace_id="tr_1", decision=decision)
    assert answer.decision == "INSUFFICIENT_EVIDENCE"
    assert answer.risk_level == "EXTREME"
    assert answer.verification.verdict == "FAIL"
    assert "outbound" in answer.limitations[0].lower()
