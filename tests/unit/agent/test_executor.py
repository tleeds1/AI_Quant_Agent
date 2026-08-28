from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel

from quantagent.agent.budget import RequestBudget
from quantagent.agent.circuit_breaker import CircuitBreakerRegistry
from quantagent.agent.executor import execute_plan
from quantagent.agent.planner import Plan, PlanStep
from quantagent.contracts.errors import ProviderUnavailableError, UnknownTickerError
from quantagent.contracts.metrics import MetricValue
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import ToolRegistry, ToolSpec
from tests.unit.agent.builders import build_provenance


class _Args(BaseModel):
    pass


class _Out(BaseModel):
    provenance: object


class _NestedOut(BaseModel):
    provenance: object
    metric: MetricValue
    sub: list[_SubModel]


class _SubModel(BaseModel):
    nested_metric: MetricValue
    label: str


_NestedOut.model_rebuild()


def _fresh_context() -> ToolContext:
    from unittest.mock import AsyncMock

    return ToolContext(
        tenant_id="tenant_1",
        portfolios=AsyncMock(),
        prices=AsyncMock(),
        fundamentals=AsyncMock(),
        factors=AsyncMock(),
        cache=AsyncMock(),
    )


def _register(
    registry: ToolRegistry,
    name: str,
    func: object,
    *,
    p95_latency_ms: int = 10,
    est_cost_usd: float = 0.0,
) -> None:
    registry._tools[name] = ToolSpec(
        name=name,
        description="d",
        p95_latency_ms=p95_latency_ms,
        est_cost_usd=est_cost_usd,
        cache_ttl_s=0,
        side_effects="READ_ONLY",
        input_model=_Args,
        output_model=_Out,
        func=func,  # type: ignore[arg-type]
    )


def _step(id_: str, tool: str, depends_on: list[str] | None = None) -> PlanStep:
    return PlanStep(id=id_, tool=tool, args={}, depends_on=depends_on or [])


async def test_independent_branches_run_concurrently() -> None:
    registry = ToolRegistry()
    windows: dict[str, tuple[float, float]] = {}

    def make_tool(name: str, delay_s: float) -> object:
        async def _tool(inp: _Args, ctx: ToolContext) -> _Out:
            start = time.monotonic()
            await asyncio.sleep(delay_s)
            windows[name] = (start, time.monotonic())
            return _Out(provenance=build_provenance(tool_call_id=f"tc_{name}"))

        return _tool

    _register(registry, "s1", make_tool("s1", 0.0))
    _register(registry, "s2", make_tool("s2", 0.05))
    _register(registry, "s5", make_tool("s5", 0.05))
    plan = Plan(
        steps=[_step("s1", "s1"), _step("s2", "s2", ["s1"]), _step("s5", "s5", ["s1"])],
        success_criteria="test",
    )
    budget = RequestBudget(max_tool_calls=12, max_wall_ms=12_000, max_usd=1.0)

    result = await execute_plan(
        plan,
        _fresh_context(),
        budget,
        "tr_test",
        registry=registry,
        circuit_breakers=CircuitBreakerRegistry(),
    )

    assert {c.status for c in result.ledger.calls} == {"OK"}
    s2_start, s2_end = windows["s2"]
    s5_start, s5_end = windows["s5"]
    # Structural overlap assertion, not a wall-clock race: holds regardless
    # of machine speed because s2/s5 are created in the SAME
    # asyncio.gather(...) call once s1 completes.
    assert s2_start < s5_end and s5_start < s2_end


async def test_single_step_plan_runs_without_special_casing() -> None:
    registry = ToolRegistry()

    async def _tool(inp: _Args, ctx: ToolContext) -> _Out:
        return _Out(provenance=build_provenance())

    _register(registry, "only", _tool)
    plan = Plan(steps=[_step("s1", "only")], success_criteria="test")
    budget = RequestBudget(max_tool_calls=12, max_wall_ms=12_000, max_usd=1.0)

    result = await execute_plan(
        plan,
        _fresh_context(),
        budget,
        "tr",
        registry=registry,
        circuit_breakers=CircuitBreakerRegistry(),
    )

    assert len(result.ledger.calls) == 1
    assert result.ledger.calls[0].status == "OK"
    assert result.budget_exhausted is False
    assert result.degraded is False


async def test_budget_exhaustion_mid_dag_returns_partial_ledger() -> None:
    registry = ToolRegistry()
    call_counts = {"s1": 0, "s2": 0, "s3": 0}

    def make_tool(name: str) -> object:
        async def _tool(inp: _Args, ctx: ToolContext) -> _Out:
            call_counts[name] += 1
            return _Out(provenance=build_provenance(tool_call_id=f"tc_{name}"))

        return _tool

    for name in ("s1", "s2", "s3"):
        _register(registry, name, make_tool(name))
    plan = Plan(
        steps=[_step("s1", "s1"), _step("s2", "s2", ["s1"]), _step("s3", "s3", ["s2"])],
        success_criteria="test",
    )
    budget = RequestBudget(max_tool_calls=1, max_wall_ms=12_000, max_usd=1.0)

    result = await execute_plan(
        plan,
        _fresh_context(),
        budget,
        "tr",
        registry=registry,
        circuit_breakers=CircuitBreakerRegistry(),
    )

    assert result.budget_exhausted is True
    assert len(result.ledger.calls) == 1
    assert call_counts == {"s1": 1, "s2": 0, "s3": 0}


async def test_timeout_produces_timeout_status_not_a_crashed_plan(monkeypatch) -> None:
    # MIN_TOOL_TIMEOUT_MS is 5000ms in production (empirically raised from an
    # initial 1000ms floor -- see executor.py's module comment); lowered
    # here so this test doesn't have to sleep 5+ real seconds.
    from quantagent.agent import executor as executor_module

    monkeypatch.setattr(executor_module, "MIN_TOOL_TIMEOUT_MS", 50)
    registry = ToolRegistry()

    async def _hangs(inp: _Args, ctx: ToolContext) -> _Out:
        await asyncio.sleep(100)
        return _Out(provenance=build_provenance())  # pragma: no cover

    _register(registry, "hangs", _hangs, p95_latency_ms=1)
    plan = Plan(steps=[_step("s1", "hangs")], success_criteria="test")
    budget = RequestBudget(max_tool_calls=12, max_wall_ms=12_000, max_usd=1.0)

    async def _run() -> None:
        result = await execute_plan(
            plan,
            _fresh_context(),
            budget,
            "tr",
            registry=registry,
            circuit_breakers=CircuitBreakerRegistry(),
        )
        assert len(result.ledger.calls) == 1
        assert result.ledger.calls[0].status == "TIMEOUT"
        assert "timeout" in (result.ledger.calls[0].error or "").lower()

    await asyncio.wait_for(_run(), timeout=5.0)


async def test_transient_failure_retries_then_succeeds() -> None:
    registry = ToolRegistry()
    state = {"calls": 0}

    async def _flaky(inp: _Args, ctx: ToolContext) -> _Out:
        state["calls"] += 1
        if state["calls"] < 3:
            raise ProviderUnavailableError("temporary blip")
        return _Out(provenance=build_provenance())

    _register(registry, "flaky", _flaky)
    plan = Plan(steps=[_step("s1", "flaky")], success_criteria="test")
    budget = RequestBudget(max_tool_calls=12, max_wall_ms=12_000, max_usd=1.0)

    result = await execute_plan(
        plan,
        _fresh_context(),
        budget,
        "tr",
        registry=registry,
        circuit_breakers=CircuitBreakerRegistry(),
    )

    assert result.ledger.calls[0].status == "OK"
    assert state["calls"] == 3


async def test_non_transient_failure_is_never_retried() -> None:
    registry = ToolRegistry()
    state = {"calls": 0}

    async def _always_bad_ticker(inp: _Args, ctx: ToolContext) -> _Out:
        state["calls"] += 1
        raise UnknownTickerError("XYZ is not a known ticker")

    _register(registry, "bad", _always_bad_ticker)
    plan = Plan(steps=[_step("s1", "bad")], success_criteria="test")
    budget = RequestBudget(max_tool_calls=12, max_wall_ms=12_000, max_usd=1.0)

    result = await execute_plan(
        plan,
        _fresh_context(),
        budget,
        "tr",
        registry=registry,
        circuit_breakers=CircuitBreakerRegistry(),
    )

    assert result.ledger.calls[0].status == "ERROR"
    assert state["calls"] == 1


async def test_circuit_breaker_opens_and_short_circuits_subsequent_calls() -> None:
    registry = ToolRegistry()
    call_count = {"n": 0}

    async def _always_unavailable(inp: _Args, ctx: ToolContext) -> _Out:
        call_count["n"] += 1
        raise ProviderUnavailableError("down")

    # All four steps map to the "prices" provider via the real tool-name
    # table (executor._TOOL_PROVIDERS) -- register the fake adapter under a
    # real provider-mapped tool name so the breaker logic actually engages.
    from quantagent.agent import executor as executor_module

    _register(registry, "get_prices", _always_unavailable)
    plan = Plan(
        steps=[
            _step("s1", "get_prices"),
            _step("s2", "get_prices"),
            _step("s3", "get_prices"),
            _step("s4", "get_prices"),
        ],
        success_criteria="test",
    )
    assert executor_module._TOOL_PROVIDERS["get_prices"] == ["prices"]
    budget = RequestBudget(max_tool_calls=12, max_wall_ms=12_000, max_usd=1.0)
    breakers = CircuitBreakerRegistry()

    result = await execute_plan(
        plan, _fresh_context(), budget, "tr", registry=registry, circuit_breakers=breakers
    )

    # 4 independent steps all dispatch in ONE wave (none depend on another),
    # so every step's circuit-breaker check happens before any of their
    # failures are recorded -- all 4 attempt the call (each retried
    # RETRY_MAX_ATTEMPTS=3 times by tenacity before _execute_step counts it
    # as one breaker failure), so 4 steps * 3 attempts = 12 real calls, even
    # though the breaker (threshold=3) ends this round OPEN.
    assert call_count["n"] == 12
    assert {c.status for c in result.ledger.calls} == {"ERROR"}

    # A second plan run against the now-open breaker must short-circuit.
    plan2 = Plan(steps=[_step("s1", "get_prices")], success_criteria="test")
    budget2 = RequestBudget(max_tool_calls=12, max_wall_ms=12_000, max_usd=1.0)
    result2 = await execute_plan(
        plan2, _fresh_context(), budget2, "tr2", registry=registry, circuit_breakers=breakers
    )
    assert result2.ledger.calls[0].status == "DEGRADED"
    assert call_count["n"] == 12  # unchanged: the call was never attempted


async def test_numeric_index_flattens_metrics_at_every_depth() -> None:
    registry = ToolRegistry()

    async def _bare_metric(inp: _Args, ctx: ToolContext) -> MetricValue:
        return MetricValue(
            metric_id="m1",
            value=1.5,
            unit="ratio",
            method="test",
            provenance=build_provenance(tool_call_id="tc_bare"),
        )

    async def _nested(inp: _Args, ctx: ToolContext) -> _NestedOut:
        return _NestedOut(
            provenance=build_provenance(tool_call_id="tc_nested"),
            metric=MetricValue(
                metric_id="m2", value=2.5, unit="pct", method="test", provenance=build_provenance()
            ),
            sub=[
                _SubModel(
                    nested_metric=MetricValue(
                        metric_id="m3",
                        value=3.5,
                        unit="usd",
                        method="test",
                        provenance=build_provenance(),
                    ),
                    label="not-a-metric",
                )
            ],
        )

    registry._tools["bare"] = ToolSpec(
        name="bare",
        description="d",
        p95_latency_ms=10,
        est_cost_usd=0.0,
        cache_ttl_s=0,
        side_effects="READ_ONLY",
        input_model=_Args,
        output_model=MetricValue,
        func=_bare_metric,  # type: ignore[arg-type]
    )
    registry._tools["nested"] = ToolSpec(
        name="nested",
        description="d",
        p95_latency_ms=10,
        est_cost_usd=0.0,
        cache_ttl_s=0,
        side_effects="READ_ONLY",
        input_model=_Args,
        output_model=_NestedOut,
        func=_nested,  # type: ignore[arg-type]
    )
    plan = Plan(steps=[_step("s1", "bare"), _step("s2", "nested")], success_criteria="test")
    budget = RequestBudget(max_tool_calls=12, max_wall_ms=12_000, max_usd=1.0)

    result = await execute_plan(
        plan,
        _fresh_context(),
        budget,
        "tr",
        registry=registry,
        circuit_breakers=CircuitBreakerRegistry(),
    )

    index = result.ledger.numeric_index
    assert index["tc_bare.result"] == 1.5
    assert index["tc_nested.result.metric"] == 2.5
    assert index["tc_nested.result.sub.0.nested_metric"] == 3.5
    assert not any("label" in key for key in index)


async def test_cache_hit_still_reports_ok_not_cached() -> None:
    """Documents the M3 scope decision: no provider surfaces a `from_cache`
    flag through `registry.invoke`, so a cache hit is indistinguishable from
    a fresh fetch at this layer -- both record "OK".
    """
    registry = ToolRegistry()

    async def _tool(inp: _Args, ctx: ToolContext) -> _Out:
        return _Out(provenance=build_provenance())

    _register(registry, "cached_tool", _tool)
    plan = Plan(steps=[_step("s1", "cached_tool")], success_criteria="test")
    budget = RequestBudget(max_tool_calls=12, max_wall_ms=12_000, max_usd=1.0)

    result = await execute_plan(
        plan,
        _fresh_context(),
        budget,
        "tr",
        registry=registry,
        circuit_breakers=CircuitBreakerRegistry(),
    )

    assert result.ledger.calls[0].status == "OK"
