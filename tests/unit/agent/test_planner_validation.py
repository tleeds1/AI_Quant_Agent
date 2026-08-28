from __future__ import annotations

from quantagent.agent.planner import Plan, PlanStep, validate_plan
from tests.unit.agent.builders import build_registry_with_tool


def _step(
    id_: str, tool: str = "dummy_tool", args: dict[str, object] | None = None, depends_on=None
) -> PlanStep:
    return PlanStep(
        id=id_, tool=tool, args=args or {"portfolio_id": "p1"}, depends_on=depends_on or []
    )


def test_empty_plan_rejected() -> None:
    errors = validate_plan(
        Plan(steps=[], success_criteria="x"), registry=build_registry_with_tool()
    )
    assert {e.code for e in errors} == {"EMPTY_PLAN"}


def test_unknown_tool_rejected() -> None:
    plan = Plan(steps=[_step("s1", tool="not_a_real_tool")], success_criteria="x")
    errors = validate_plan(plan, registry=build_registry_with_tool())
    assert any(e.code == "UNKNOWN_TOOL" and e.step_id == "s1" for e in errors)


def test_invalid_args_rejected() -> None:
    plan = Plan(steps=[_step("s1", args={"wrong_field": 1})], success_criteria="x")
    errors = validate_plan(plan, registry=build_registry_with_tool())
    assert any(e.code == "INVALID_ARGS" and e.step_id == "s1" for e in errors)


def test_self_dependency_rejected() -> None:
    plan = Plan(steps=[_step("s1", depends_on=["s1"])], success_criteria="x")
    errors = validate_plan(plan, registry=build_registry_with_tool())
    assert any(e.code == "SELF_DEPENDENCY" for e in errors)


def test_dangling_dependency_rejected() -> None:
    plan = Plan(steps=[_step("s1", depends_on=["s_missing"])], success_criteria="x")
    errors = validate_plan(plan, registry=build_registry_with_tool())
    assert any(e.code == "DANGLING_DEPENDENCY" for e in errors)


def test_cycle_rejected() -> None:
    plan = Plan(
        steps=[_step("s1", depends_on=["s2"]), _step("s2", depends_on=["s1"])],
        success_criteria="x",
    )
    errors = validate_plan(plan, registry=build_registry_with_tool())
    assert any(e.code == "CYCLE" for e in errors)


def test_duplicate_step_ids_rejected() -> None:
    plan = Plan(steps=[_step("s1"), _step("s1")], success_criteria="x")
    errors = validate_plan(plan, registry=build_registry_with_tool())
    assert any(e.code == "DUPLICATE_STEP_ID" for e in errors)


def test_too_many_steps_rejected() -> None:
    plan = Plan(steps=[_step(f"s{i}") for i in range(13)], success_criteria="x")
    errors = validate_plan(plan, registry=build_registry_with_tool(), max_steps=12)
    assert any(e.code == "TOO_MANY_STEPS" for e in errors)


def test_latency_budget_exceeded_uses_critical_path_not_flat_sum() -> None:
    # Two independent 300ms branches -> critical path is 300ms (parallel),
    # not 600ms (a flat sum would wrongly reject a plan that fits the budget).
    registry = build_registry_with_tool(p95_latency_ms=300)
    plan = Plan(steps=[_step("s1"), _step("s2")], success_criteria="x")

    assert validate_plan(plan, registry=registry, max_wall_ms=400) == []
    errors = validate_plan(plan, registry=registry, max_wall_ms=200)
    assert any(e.code == "BUDGET_LATENCY_EXCEEDED" for e in errors)


def test_latency_budget_sums_along_a_dependency_chain() -> None:
    registry = build_registry_with_tool(p95_latency_ms=300)
    plan = Plan(steps=[_step("s1"), _step("s2", depends_on=["s1"])], success_criteria="x")

    assert validate_plan(plan, registry=registry, max_wall_ms=700) == []
    errors = validate_plan(plan, registry=registry, max_wall_ms=500)
    assert any(e.code == "BUDGET_LATENCY_EXCEEDED" for e in errors)


def test_cost_budget_exceeded_rejected() -> None:
    # All 18 real tools carry est_cost_usd=0.0 today -- this proves the
    # check is real code, not a stub, via a fabricated nonzero-cost tool.
    registry = build_registry_with_tool(est_cost_usd=0.10)
    plan = Plan(steps=[_step("s1"), _step("s2")], success_criteria="x")

    errors = validate_plan(plan, registry=registry, max_usd=0.15)
    assert any(e.code == "BUDGET_COST_EXCEEDED" for e in errors)


def test_valid_plan_passes() -> None:
    plan = Plan(steps=[_step("s1"), _step("s2", depends_on=["s1"])], success_criteria="x")
    assert validate_plan(plan, registry=build_registry_with_tool()) == []


def test_dangling_dependency_suppresses_cycle_check_to_avoid_key_error() -> None:
    # A dangling ref alongside what would otherwise look cyclic must not
    # crash -- validate_plan must not attempt the cycle walk over an
    # ill-formed graph.
    plan = Plan(steps=[_step("s1", depends_on=["s_missing"])], success_criteria="x")
    errors = validate_plan(plan, registry=build_registry_with_tool())
    assert {e.code for e in errors} == {"DANGLING_DEPENDENCY"}
