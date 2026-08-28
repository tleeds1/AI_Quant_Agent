from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel

from quantagent.contracts.provenance import Provenance
from quantagent.tools.registry import ToolRegistry, ToolSpec


class _FakeInput(BaseModel):
    portfolio_id: str


class _FakeOutput(BaseModel):
    portfolio_id: str
    provenance: Provenance


async def _fake_adapter(inp: _FakeInput, ctx: object) -> _FakeOutput:  # pragma: no cover
    raise NotImplementedError("register a real func for tests that actually invoke this")


def build_provenance(tool_call_id: str = "tc_test", **overrides: object) -> Provenance:
    defaults: dict[str, object] = dict(
        tool_call_id=tool_call_id,
        tool_name="fake_tool",
        as_of=date(2026, 8, 22),
        computed_at=datetime(2026, 8, 22, 12, 0, 0),
        inputs_hash="fakehash",
        data_sources=["fake"],
        estimator=None,
        sample_size=None,
        seed=None,
        warnings=[],
    )
    defaults.update(overrides)
    return Provenance(**defaults)  # type: ignore[arg-type]


def build_registry_with_tool(
    name: str = "dummy_tool",
    *,
    p95_latency_ms: int = 100,
    est_cost_usd: float = 0.0,
    input_model: type[BaseModel] = _FakeInput,
    output_model: type[BaseModel] = _FakeOutput,
    func: object = _fake_adapter,
) -> ToolRegistry:
    """A fresh, local `ToolRegistry()` (never the process-wide singleton),
    mirroring `tests/unit/tools/test_registry.py`'s own convention, with one
    tool directly constructed (bypassing the `@registry.tool(...)` decorator,
    which would otherwise require `func`'s real signature to introspect).
    """
    registry = ToolRegistry()
    registry._tools[name] = ToolSpec(  # test-only direct construction
        name=name,
        description="a fake tool for tests",
        p95_latency_ms=p95_latency_ms,
        est_cost_usd=est_cost_usd,
        cache_ttl_s=0,
        side_effects="READ_ONLY",
        input_model=input_model,
        output_model=output_model,
        func=func,  # type: ignore[arg-type]
    )
    return registry
