from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import signature
from typing import Any, Literal, TypeVar, cast

from pydantic import BaseModel, ValidationError

from quantagent.contracts.errors import ToolValidationError
from quantagent.data.cache import compute_inputs_hash
from quantagent.tools.context import ToolContext

AdapterFunc = Callable[[BaseModel, ToolContext], Awaitable[BaseModel]]
# Deliberately unbound to AdapterFunc: each concrete adapter's parameter type
# (e.g. `ComputeExpressionInput`, not `BaseModel`) is narrower than
# `AdapterFunc`'s, which function contravariance correctly rejects as an
# `AdapterFunc` itself. `ToolSpec.func` erases to `AdapterFunc` deliberately
# (the registry really does receive untyped `dict[str, Any]` args at
# runtime) -- the erasure is a documented `cast` at the one storage site
# below, not a hole in every adapter's own signature.
F = TypeVar("F", bound=Callable[..., Awaitable[BaseModel]])


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    p95_latency_ms: int
    est_cost_usd: float
    cache_ttl_s: int
    side_effects: Literal["READ_ONLY"]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    func: AdapterFunc

    def json_schema(self) -> dict[str, Any]:
        """Schema handed to the LLM/MCP client -- generated, never hand-written
        (guideline.md §5 rule 4).
        """
        return self.input_model.model_json_schema()


class ToolRegistry:
    """Process-wide registry populated by `@registry.tool(...)` decorators at
    import time. `tools/__init__.py` imports every tool module so decoration
    runs before `list_tools()`/`invoke()` is ever called.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def tool(
        self,
        *,
        name: str,
        description: str,
        p95_latency_ms: int,
        est_cost_usd: float,
        cache_ttl_s: int,
        side_effects: Literal["READ_ONLY"] = "READ_ONLY",
    ) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            if name in self._tools:
                raise ValueError(f"tool {name!r} is already registered")
            input_model, output_model = _extract_io_models(func)
            self._tools[name] = ToolSpec(
                name=name,
                description=description,
                p95_latency_ms=p95_latency_ms,
                est_cost_usd=est_cost_usd,
                cache_ttl_s=cache_ttl_s,
                side_effects=side_effects,
                input_model=input_model,
                output_model=output_model,
                func=cast(AdapterFunc, func),
            )
            return func

        return decorator

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    async def invoke(self, name: str, raw_args: dict[str, Any], ctx: ToolContext) -> BaseModel:
        spec = self._tools.get(name)
        if spec is None:
            raise ToolValidationError(f"unknown tool {name!r}")
        try:
            validated = spec.input_model.model_validate(raw_args)
        except ValidationError as exc:
            raise ToolValidationError(f"invalid arguments for tool {name!r}: {exc}") from exc

        inputs_hash = compute_inputs_hash(tool_name=name, **validated.model_dump(mode="json"))
        bound_ctx = ctx.for_call(tool_name=name, inputs_hash=inputs_hash)
        return await spec.func(validated, bound_ctx)


def _extract_io_models(
    func: Callable[..., Awaitable[BaseModel]],
) -> tuple[type[BaseModel], type[BaseModel]]:
    """Reads the adapter's `(inp: XInput, ctx: ToolContext) -> XOutput`
    signature. Uses `eval_str=True` because `from __future__ import
    annotations` makes every annotation on `func` a lazy string; without
    `eval_str=True`, `params[0].annotation` would be the literal string
    `"XInput"`, not the class, and `.model_json_schema()` would blow up with
    a confusing AttributeError far from this line.
    """
    sig = signature(func, eval_str=True)
    params = list(sig.parameters.values())
    if len(params) != 2:
        raise TypeError(f"tool adapter {func.__name__!r} must take exactly (inp, ctx)")
    input_model = params[0].annotation
    output_model = sig.return_annotation
    if not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
        raise TypeError(f"tool adapter {func.__name__!r}'s input must be a BaseModel subclass")
    if not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
        raise TypeError(f"tool adapter {func.__name__!r}'s output must be a BaseModel subclass")
    return input_model, output_model


registry = ToolRegistry()
