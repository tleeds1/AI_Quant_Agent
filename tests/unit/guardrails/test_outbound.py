"""tests/unit/guardrails/test_outbound.py"""

from __future__ import annotations

import pytest

from quantagent.contracts.errors import PolicyViolationError
from quantagent.guardrails.outbound import (
    apply_disclosures,
    check_advice_framing,
    check_leakage,
    check_pii_egress,
    check_prohibited_language,
    enforce_outbound,
)
from tests.unit.guardrails.builders import build_context, build_outbound_payload
from tests.unit.verify.builders import build_answer, build_evidence


def test_prohibited_language_blocks_guarantee_claim() -> None:
    payload = build_outbound_payload(summary="This portfolio is guaranteed to outperform.")
    decision = check_prohibited_language(payload, build_context())
    assert decision.action == "BLOCK"
    assert decision.category == "prohibited_language"


def test_prohibited_language_allows_hedged_summary() -> None:
    payload = build_outbound_payload(summary="Portfolio VaR is 2.4% based on historical data.")
    decision = check_prohibited_language(payload, build_context())
    assert decision.action == "ALLOW"


def test_advice_framing_blocks_unlicensed_suitability_language() -> None:
    payload = build_outbound_payload(summary="You should buy more NVDA right now.")
    decision = check_advice_framing(payload, build_context())
    assert decision.action == "BLOCK"
    assert decision.category == "advice_framing"


def test_advice_framing_allows_licensed_tenant() -> None:
    payload = build_outbound_payload(summary="You should buy more NVDA right now.")
    context = build_context(licensed_for_suitability_advice=True)
    decision = check_advice_framing(payload, context)
    assert decision.action == "ALLOW"


def test_pii_egress_blocks_email_in_summary() -> None:
    payload = build_outbound_payload(summary="Contact jane.doe@example.com for details.")
    decision = check_pii_egress(payload, build_context())
    assert decision.action == "BLOCK"
    assert decision.category == "pii_egress"


def test_pii_egress_allows_clean_summary() -> None:
    payload = build_outbound_payload(summary="Portfolio VaR is 2.4%.")
    decision = check_pii_egress(payload, build_context())
    assert decision.action == "ALLOW"


def test_pii_egress_checks_evidence_excerpts() -> None:
    payload = build_outbound_payload(
        evidence=[build_evidence(excerpt="Contact jane.doe@example.com for the filing.")]
    )
    decision = check_pii_egress(payload, build_context())
    assert decision.action == "BLOCK"


def test_leakage_blocks_api_key_shaped_string() -> None:
    payload = build_outbound_payload(
        summary="Here is a debug value: sk-ant-abcdefghijklmnopqrstuvwxyz012345"
    )
    decision = check_leakage(payload, build_context())
    assert decision.action == "BLOCK"
    assert decision.category == "leakage"


def test_leakage_allows_clean_summary() -> None:
    payload = build_outbound_payload(summary="Portfolio VaR is 2.4%.")
    decision = check_leakage(payload, build_context())
    assert decision.action == "ALLOW"


def test_enforce_outbound_raises_policy_violation_error() -> None:
    payload = build_outbound_payload(summary="This is guaranteed to never lose money.")
    with pytest.raises(PolicyViolationError):
        enforce_outbound(payload, build_context())


def test_enforce_outbound_allows_clean_answer() -> None:
    payload = build_outbound_payload(summary="Portfolio VaR is 2.4%.")
    enforce_outbound(payload, build_context())  # must not raise


def test_apply_disclosures_replaces_with_mandatory_set() -> None:
    answer = build_answer(disclosures=["some LLM-authored disclosure text"])
    updated = apply_disclosures(answer)
    assert len(updated.disclosures) == 4
    assert "some LLM-authored disclosure text" not in updated.disclosures


def test_apply_disclosures_is_idempotent() -> None:
    answer = build_answer(disclosures=[])
    once = apply_disclosures(answer)
    twice = apply_disclosures(once)
    assert twice.disclosures == once.disclosures


def test_apply_disclosures_adds_simulated_scenario_when_hypothetical_mentioned() -> None:
    answer = build_answer(
        disclosures=[], summary="Under a hypothetical AI-basket drawdown, VaR would rise."
    )
    updated = apply_disclosures(answer)
    assert any("hypothetical" in d.lower() or "simulated" in d.lower() for d in updated.disclosures)


def test_apply_disclosures_does_not_touch_summary_or_claims() -> None:
    answer = build_answer(disclosures=[], summary="Portfolio VaR is 2.4%.")
    updated = apply_disclosures(answer)
    assert updated.summary == answer.summary
    assert updated.claims == answer.claims
