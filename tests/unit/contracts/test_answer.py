from __future__ import annotations

import pytest
from pydantic import ValidationError

from quantagent.contracts.answer import AgentAnswer
from tests.unit.contracts.test_evidence import build_claim, build_evidence
from tests.unit.contracts.test_metrics import build_metric_value
from tests.unit.contracts.test_verification import (
    build_constraint_check,
    build_verification_report,
)


def build_agent_answer(**overrides: object) -> AgentAnswer:
    """Fixture modeled on the worked example in architecture.md §16."""
    defaults: dict[str, object] = {
        "trace_id": "tr_9f3c1a",
        "scope": "PORTFOLIO",
        "decision": "REDUCE",
        "confidence": 0.65,
        "confidence_basis": ["theme_estimator_spread_13pp -> cap 0.65"],
        "risk_level": "HIGH",
        "horizon": "1-3 months",
        "summary": (
            "Yes - on every measure your AI exposure sits above your 25% thematic cap, "
            "and the concentration is more severe in risk terms than in weight terms."
        ),
        "claims": [build_claim()],
        "evidence": [build_evidence()],
        "quant_metrics": {"portfolio_var_95_1d": build_metric_value()},
        "constraints_checked": [build_constraint_check()],
        "limitations": [
            "Theme estimators disagree by 13pp (31%-44%); true exposure is uncertain "
            "within that band."
        ],
        "disclosures": ["Analysis only, not investment advice."],
        "verification": build_verification_report(),
    }
    defaults.update(overrides)
    return AgentAnswer(**defaults)


def test_agent_answer_round_trips_through_json_with_full_nested_graph() -> None:
    original = build_agent_answer()

    restored = AgentAnswer.model_validate_json(original.model_dump_json())

    assert restored == original


def test_agent_answer_rejects_empty_limitations() -> None:
    with pytest.raises(ValidationError):
        build_agent_answer(limitations=[])
