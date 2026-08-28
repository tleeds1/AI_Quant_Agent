from __future__ import annotations

import pytest
from pydantic import BaseModel

from quantagent.contracts.errors import ToolValidationError
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import ToolRegistry
from tests.unit.tools.builders import build_tool_context


class _SpyInput(BaseModel):
    value: int


class _SpyOutput(BaseModel):
    doubled: int


def _build_registry_with_spy() -> tuple[ToolRegistry, list[ToolContext]]:
    registry = ToolRegistry()
    seen_contexts: list[ToolContext] = []

    @registry.tool(
        name="spy_tool",
        description="use for testing / do not use in production",
        p95_latency_ms=1,
        est_cost_usd=0.0,
        cache_ttl_s=0,
    )
    async def spy_tool(inp: _SpyInput, ctx: ToolContext) -> _SpyOutput:
        seen_contexts.append(ctx)
        return _SpyOutput(doubled=inp.value * 2)

    return registry, seen_contexts


def test_tool_decorator_registers_under_the_given_name() -> None:
    registry, _ = _build_registry_with_spy()

    spec = registry.get("spy_tool")

    assert spec is not None
    assert spec.name == "spy_tool"
    assert spec.input_model is _SpyInput
    assert spec.output_model is _SpyOutput


def test_registering_a_duplicate_name_raises_value_error() -> None:
    registry, _ = _build_registry_with_spy()

    with pytest.raises(ValueError):

        @registry.tool(
            name="spy_tool",
            description="duplicate",
            p95_latency_ms=1,
            est_cost_usd=0.0,
            cache_ttl_s=0,
        )
        async def duplicate(inp: _SpyInput, ctx: ToolContext) -> _SpyOutput:
            return _SpyOutput(doubled=0)


def test_list_tools_returns_every_registered_tool() -> None:
    registry, _ = _build_registry_with_spy()

    specs = registry.list_tools()

    assert [s.name for s in specs] == ["spy_tool"]


def test_json_schema_is_generated_from_the_input_model() -> None:
    registry, _ = _build_registry_with_spy()

    schema = registry.get("spy_tool").json_schema()  # type: ignore[union-attr]

    assert schema == _SpyInput.model_json_schema()


async def test_invoke_raises_on_unknown_tool_name() -> None:
    registry, _ = _build_registry_with_spy()
    ctx = build_tool_context()

    with pytest.raises(ToolValidationError):
        await registry.invoke("nonexistent_tool", {}, ctx)


async def test_invoke_raises_on_invalid_arguments() -> None:
    registry, _ = _build_registry_with_spy()
    ctx = build_tool_context()

    with pytest.raises(ToolValidationError):
        await registry.invoke("spy_tool", {"value": "not an int"}, ctx)


async def test_invoke_calls_the_adapter_with_a_call_bound_context() -> None:
    registry, seen_contexts = _build_registry_with_spy()
    ctx = build_tool_context()

    result = await registry.invoke("spy_tool", {"value": 21}, ctx)

    assert result.doubled == 42
    assert len(seen_contexts) == 1
    bound_ctx = seen_contexts[0]
    assert bound_ctx.build_provenance().tool_name == "spy_tool"
    assert bound_ctx is not ctx  # call-scoped, not the shared context itself
