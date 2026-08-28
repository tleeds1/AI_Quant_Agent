"""agent/executor.py -- the async DAG executor (architecture.md §4.2:
"independent branches run in parallel via asyncio; per-tool timeout; retry
w/ jittered backoff for transient errors only; circuit breaker per provider;
results land in an append-only ledger, sole source of truth downstream.").
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol, cast
from uuid import uuid4

import structlog
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from quantagent.agent.budget import RequestBudget
from quantagent.agent.circuit_breaker import CircuitBreakerRegistry, provider_circuit_breakers
from quantagent.agent.planner import Plan, PlanStep
from quantagent.contracts.errors import ProviderUnavailableError
from quantagent.contracts.ledger import Ledger, ToolCallRecord, ToolCallStatus
from quantagent.contracts.metrics import MetricValue
from quantagent.contracts.provenance import Provenance
from quantagent.data.cache import compute_inputs_hash
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import ToolRegistry
from quantagent.tools.registry import registry as tools_registry

logger = structlog.get_logger(__name__)

# p95_latency_ms is a declared plan-time ESTIMATE (architecture.md §3.4), not
# a live measurement; real tail latency routinely exceeds it. 3x gives real
# headroom over the declared estimate. MIN_TOOL_TIMEOUT_MS floors it --
# empirically verified against LIVE yfinance during M3 implementation
# (uncached get_holdings/get_sector_exposure/calculate_portfolio_var/
# calculate_component_var calls against the real seeded pf_demo portfolio
# all exceeded a 1000ms floor and TIMED OUT, even though their declared
# p95_latency_ms values -- 400/1200/250/350ms -- are M2's own estimates, not
# redesigned here). Cold, uncached third-party network calls (yfinance's
# get_info() in particular is well known to take multiple seconds) routinely
# exceed a 1s floor; 5000ms was chosen as a floor that let every one of
# those four real calls succeed on re-verification, still comfortably below
# max_wall_ms=12000 for a single step. This is an empirically-informed
# adjustment to THIS module's own tunable constant, not a change to M2's
# frozen ToolSpec.p95_latency_ms declarations -- those may be worth
# recalibrating in a future pass with real production telemetry (M6
# observability scope), flagged here rather than silently guessed at twice.
TOOL_TIMEOUT_MULTIPLIER = 3
MIN_TOOL_TIMEOUT_MS = 5000

# Applies only to ProviderUnavailableError -- every other DataError subtype
# (UnknownTickerError, InsufficientDataError, StaleDataError) is
# deterministic given the same args and must NEVER be retried. The retry
# sequence, including backoff sleeps, runs INSIDE the single per-tool
# asyncio.timeout() below rather than getting its own separate ceiling: one
# wall-clock bound per step is simpler to reason about against
# max_wall_ms=12_000 than two stacked ones.
RETRY_MAX_ATTEMPTS = 3
RETRY_BACKOFF_BASE_S = 0.1
RETRY_BACKOFF_MAX_S = 1.0

# Provider identity as a static table owned entirely by this module, not a
# field on tools.registry.ToolSpec: a facility only the executor needs has
# no business widening the finished, tested M2 registry contract. A tool
# may depend on more than one provider (e.g. get_sector_exposure needs both
# prices and fundamentals); if ANY of a tool's declared providers has an
# open circuit, the call is short-circuited, since it would fail anyway.
_TOOL_PROVIDERS: dict[str, list[str]] = {
    "get_prices": ["prices"],
    "get_returns": ["prices"],
    "get_correlation_matrix": ["prices"],
    "get_concentration_metrics": ["prices"],
    "get_holdings": ["prices"],
    "calculate_portfolio_var": ["prices"],
    "calculate_cvar": ["prices"],
    "calculate_component_var": ["prices"],
    "calculate_max_drawdown": ["prices"],
    "get_portfolio_beta": ["prices"],
    "calculate_tracking_error": ["prices"],
    "generate_risk_report": ["prices"],
    "get_fundamentals": ["fundamentals"],
    "get_sector_exposure": ["prices", "fundamentals"],
    "get_factor_exposure": ["prices", "factors"],
    "get_portfolio": [],
    "get_transactions": [],
    "compute_expression": [],
}


class _HasProvenance(Protocol):
    """Every tool output carries a top-level `provenance: Provenance` field
    (I7 invariant), but that isn't encoded in `registry.invoke`'s generic
    `BaseModel` return type -- this Protocol names the one attribute this
    module actually needs, cast at the single site below, mirroring
    `tools/registry.py`'s own documented single-cast-site precedent.
    """

    provenance: Provenance


@dataclass(slots=True)
class ExecutionResult:
    """Handed to the loop's state machine after EXECUTE. `budget_exhausted`
    and `degraded` are the two signals the DEGRADE/repair-gate transitions
    need (architecture.md §4.2's diagram); finer detail (which provider,
    which step) is available on `ledger.calls` directly.
    """

    ledger: Ledger
    budget_exhausted: bool
    degraded: bool


@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_random_exponential(multiplier=RETRY_BACKOFF_BASE_S, max=RETRY_BACKOFF_MAX_S),
    retry=retry_if_exception_type(ProviderUnavailableError),
    reraise=True,
)
async def _invoke_with_retry(
    registry: ToolRegistry, tool: str, args: dict[str, Any], ctx: ToolContext
) -> BaseModel:
    return await registry.invoke(tool, args, ctx)


def _record_breaker_outcome(
    breakers: CircuitBreakerRegistry, providers: list[str], *, success: bool
) -> None:
    for provider in providers:
        breaker = breakers.get(provider)
        if success:
            breaker.record_success()
        else:
            breaker.record_failure()


def _flatten_metric_values(prefix: str, obj: Any) -> dict[str, float]:
    """Structural walk of a tool's OWN typed output model (called before
    `model_dump()`, not a shape-sniff of the dumped dict afterward) for
    every `MetricValue`, however deeply nested in lists/dicts/sub-models.
    """
    found: dict[str, float] = {}
    if isinstance(obj, MetricValue):
        found[prefix] = obj.value
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            found.update(_flatten_metric_values(f"{prefix}.{name}", getattr(obj, name)))
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            found.update(_flatten_metric_values(f"{prefix}.{idx}", item))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            found.update(_flatten_metric_values(f"{prefix}.{key}", value))
    return found


def _failed_record(step: PlanStep, *, status: ToolCallStatus, error: str) -> ToolCallRecord:
    """Shared constructor for ERROR/TIMEOUT/DEGRADED records, where no
    `Provenance` was ever minted (the call never completed successfully),
    so `call_id`/`args_hash` are synthesized locally.
    """
    return ToolCallRecord(
        call_id=f"tc_{uuid4().hex[:10]}",
        tool_name=step.tool,
        args=step.args,
        args_hash=compute_inputs_hash(tool_name=step.tool, **step.args),
        status=status,
        latency_ms=0,
        cost_usd=0.0,
        result=None,
        error=error,
    )


async def _execute_step(
    step: PlanStep,
    ctx: ToolContext,
    registry: ToolRegistry,
    breakers: CircuitBreakerRegistry,
    trace_id: str,
) -> tuple[ToolCallRecord, dict[str, float]]:
    """Executes one DAG step to completion (including retries) and NEVER
    raises: every outcome becomes an explicit `ToolCallRecord`. This is what
    lets `execute_plan` use plain `asyncio.gather` -- nothing is swallowed,
    every failure is recorded and visible downstream.
    """
    spec = registry.get(step.tool)
    if spec is None:
        # Unreachable when the plan came from validate_plan, but a fresh
        # ToolRegistry passed by a caller/test could still name an unknown
        # tool -- fail this one step loudly rather than crash the whole DAG.
        return _failed_record(step, status="ERROR", error=f"unknown tool {step.tool!r}"), {}

    providers = _TOOL_PROVIDERS.get(step.tool, [])
    blocked = [p for p in providers if not breakers.get(p).try_reserve_call()]
    if blocked:
        logger.warning(
            "tool_call_degraded_circuit_open", trace_id=trace_id, tool=step.tool, providers=blocked
        )
        return (
            _failed_record(
                step, status="DEGRADED", error=f"circuit open for provider(s): {', '.join(blocked)}"
            ),
            {},
        )

    timeout_s = max(spec.p95_latency_ms * TOOL_TIMEOUT_MULTIPLIER, MIN_TOOL_TIMEOUT_MS) / 1000
    start = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s):
            result = await _invoke_with_retry(registry, step.tool, step.args, ctx)
    except TimeoutError:
        logger.warning("tool_call_timeout", trace_id=trace_id, tool=step.tool, timeout_s=timeout_s)
        return (
            _failed_record(
                step, status="TIMEOUT", error=f"exceeded {timeout_s * 1000:.0f}ms timeout"
            ),
            {},
        )
    except ProviderUnavailableError as exc:
        # Retries already exhausted inside _invoke_with_retry -- a genuine,
        # persistent provider-availability failure, so it counts against
        # the circuit breaker.
        _record_breaker_outcome(breakers, providers, success=False)
        logger.warning(
            "tool_call_provider_unavailable", trace_id=trace_id, tool=step.tool, error=str(exc)
        )
        return _failed_record(step, status="ERROR", error=str(exc)), {}
    except Exception as exc:
        # Non-transient business-logic failure (bad ticker, insufficient
        # data) or an unexpected bug -- not a provider-health signal, so the
        # breaker is left untouched.
        logger.warning("tool_call_error", trace_id=trace_id, tool=step.tool, error=str(exc))
        return _failed_record(step, status="ERROR", error=str(exc)), {}

    latency_ms = int((time.monotonic() - start) * 1000)
    _record_breaker_outcome(breakers, providers, success=True)
    provenance = cast(_HasProvenance, result).provenance
    record = ToolCallRecord(
        call_id=provenance.tool_call_id,
        tool_name=step.tool,
        args=step.args,
        args_hash=provenance.inputs_hash,
        status="OK",  # M3 never distinguishes CACHED -- see module docstring below
        latency_ms=latency_ms,
        cost_usd=spec.est_cost_usd,
        result=result.model_dump(mode="json"),
        error=None,
    )
    contributed = _flatten_metric_values(f"{record.call_id}.result", result)
    return record, contributed


async def execute_plan(
    plan: Plan,
    ctx: ToolContext,
    budget: RequestBudget,
    trace_id: str,
    *,
    registry: ToolRegistry = tools_registry,
    circuit_breakers: CircuitBreakerRegistry = provider_circuit_breakers,
) -> ExecutionResult:
    """Runs `plan.steps` in topological waves: every step whose
    `depends_on` are all complete is dispatched in the SAME `asyncio.gather`
    call as every other simultaneously-ready step, so independent branches
    genuinely run concurrently.

    Uses `asyncio.gather`, not `asyncio.TaskGroup`: every step coroutine
    (`_execute_step`) is written to never raise, so `TaskGroup`'s
    all-or-nothing cancel-siblings-on-first-exception behaviour would never
    trigger for step failures -- the one thing that would matter for is
    budget exhaustion, and exhaustion isn't a per-task exception here, it's
    a plain boolean checked between waves.

    Each wave is clipped to `budget.max_tool_calls - budget.calls_made`
    steps before dispatch, so the call-count budget can never be exceeded;
    `max_wall_ms`/`max_usd` are re-checked before every wave, so an
    in-flight wave can overshoot by at most one wave's worth before the next
    wave is refused -- the documented, graceful (not exact) degradation the
    DoD asks for.

    Never re-validates the plan (`validate_plan` already guarantees no
    cycles, no unknown tools, no dangling/duplicate ids). A single-step,
    zero-dependency plan (the DIRECT_TOOL fast path) runs through the
    identical wave loop as any other plan: one wave of size 1, then the loop
    terminates -- no special casing needed or present.

    `status="CACHED"` is a real value in `ToolCallRecord.status`'s Literal
    but is never produced here: none of `data/providers/*`'s cache-aside
    implementations surface a `from_cache` flag anywhere `registry.invoke`'s
    return value exposes it, and adding one is an M2 plumbing change out of
    this milestone's scope. Every successful call is recorded "OK"
    regardless of whether the underlying provider served it from Redis.
    """
    steps_by_id = {s.id: s for s in plan.steps}
    plan_order = {s.id: i for i, s in enumerate(plan.steps)}
    pending_ids = set(steps_by_id)
    completed_ids: set[str] = set()
    calls: list[ToolCallRecord] = []
    numeric_index: dict[str, float] = {}
    budget_exhausted = False

    while pending_ids:
        if not budget.has_remaining_capacity():
            budget_exhausted = True
            break

        ready_ids = sorted(
            (sid for sid in pending_ids if set(steps_by_id[sid].depends_on) <= completed_ids),
            key=lambda sid: plan_order[sid],
        )
        if not ready_ids:
            # Unreachable given validate_plan's guarantees; a defensive,
            # loud failure rather than a silent truncation if that contract
            # is ever violated upstream (e.g. a hand-built test plan).
            raise RuntimeError(
                f"execute_plan: {len(pending_ids)} step(s) left with no dispatchable step -- "
                "validate_plan should have rejected this plan"
            )

        slots_left = budget.max_tool_calls - budget.calls_made
        wave_ids = ready_ids[:slots_left]
        wave_steps = [steps_by_id[sid] for sid in wave_ids]

        results = await asyncio.gather(
            *(_execute_step(s, ctx, registry, circuit_breakers, trace_id) for s in wave_steps)
        )
        for step, (record, contributed) in zip(wave_steps, results, strict=True):
            calls.append(record)
            numeric_index.update(contributed)
            budget.record_call(record)
            completed_ids.add(step.id)
            pending_ids.discard(step.id)

    ledger = Ledger(trace_id=trace_id, calls=calls, numeric_index=numeric_index)
    return ExecutionResult(
        ledger=ledger,
        budget_exhausted=budget_exhausted,
        degraded=any(c.status == "DEGRADED" for c in calls),
    )
