from __future__ import annotations

from quantagent.agent.budget import RequestBudget
from quantagent.contracts.ledger import ToolCallRecord


def _record(**overrides: object) -> ToolCallRecord:
    defaults: dict[str, object] = dict(
        call_id="tc_1",
        tool_name="dummy",
        args={},
        args_hash="h",
        status="OK",
        latency_ms=10,
        cost_usd=0.0,
        result=None,
        error=None,
    )
    defaults.update(overrides)
    return ToolCallRecord(**defaults)  # type: ignore[arg-type]


def test_has_remaining_capacity_true_initially() -> None:
    budget = RequestBudget(max_tool_calls=2, max_wall_ms=10_000, max_usd=1.0)
    assert budget.has_remaining_capacity() is True


def test_call_count_exhaustion() -> None:
    budget = RequestBudget(max_tool_calls=1, max_wall_ms=10_000, max_usd=1.0)
    budget.record_call(_record())
    assert budget.has_remaining_capacity() is False


def test_cost_exhaustion() -> None:
    budget = RequestBudget(max_tool_calls=10, max_wall_ms=10_000, max_usd=0.05)
    budget.record_call(_record(cost_usd=0.10))
    assert budget.has_remaining_capacity() is False


def test_degraded_call_still_counts_against_call_budget() -> None:
    budget = RequestBudget(max_tool_calls=1, max_wall_ms=10_000, max_usd=1.0)
    budget.record_call(_record(status="DEGRADED", cost_usd=0.0))
    assert budget.has_remaining_capacity() is False


def test_limitation_text_is_stable() -> None:
    assert "budget" in RequestBudget.limitation_text().lower()


def test_from_settings_constructs_from_config_defaults() -> None:
    budget = RequestBudget.from_settings()
    assert budget.max_tool_calls > 0
    assert budget.max_wall_ms > 0
    assert budget.max_usd > 0
