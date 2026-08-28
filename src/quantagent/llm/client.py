"""llm/client.py -- the one shared structured-output primitive every LLM-
calling stage (INTAKE, PLAN, SYNTHESIZE) builds on (guideline.md §7:
"Structured output via tool/function schema, not 'respond in JSON'. Parse
with Pydantic and retry once on validation failure with the validation error
appended.").

Verified against the installed `anthropic==1.0.0` SDK (not assumed from
older tutorials, per this project's established discipline -- see the M2
MCP-SDK precedent): tool-forcing via `tools`/`tool_choice={"type":"tool",
"name":...}` is unchanged and used here. **`temperature` no longer exists
anywhere in this SDK's request surface** (confirmed: zero references to
"temperature" in the entire installed `anthropic` package) -- it has been
replaced by `output_config.effort` (`"low"|"medium"|"high"|"xhigh"|"max"`),
a reasoning-effort control, not a sampling-temperature one. guideline.md §7
prescribes `temperature=0` for planning/verification/classification and
`<=0.3` for synthesis; since no literal temperature knob is available
against this SDK/API version, `get_structured_completion` keeps
`temperature` as its own public parameter (so every call site -- INTAKE,
PLAN, SYNTHESIZE -- keeps using the vocabulary guideline.md §7 specifies)
and translates it internally to `effort` via `_effort_for_temperature`: this
is a documented, flagged adaptation, not a silent guess, and it is
inherently untestable against a live model in this environment (no real
`ANTHROPIC_API_KEY` is configured) -- revisit if a future SDK reintroduces
`temperature` or documents an official migration mapping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import structlog
from anthropic import AsyncAnthropic
from anthropic.types import (
    Message,
    MessageParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)
from anthropic.types.output_config_param import OutputConfigParam
from pydantic import BaseModel, ValidationError

from quantagent.contracts.errors import StructuredOutputError
from quantagent.llm.pricing import estimate_cost_usd

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_OUTPUT_TOOL_NAME = "emit_structured_output"
_DEFAULT_MAX_TOKENS = 2048


@dataclass(frozen=True, slots=True)
class LLMCallMetadata:
    """Per-call trace record (architecture.md §9.1: "all LLM calls: prompt
    hash, model, tokens, cost, latency"). `retried` is `True` iff the first
    response failed schema validation and a second attempt was made,
    regardless of whether the second attempt itself succeeded.
    """

    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    retried: bool


def _effort_for_temperature(temperature: float) -> Literal["low", "medium", "high"]:
    """See module docstring: the closest available analogue to guideline.md
    §7's temperature policy against an SDK with no `temperature` parameter.
    """
    if temperature <= 0.05:
        return "high"
    if temperature <= 0.3:
        return "medium"
    return "low"


def _tool_param(output_schema: type[BaseModel]) -> ToolParam:
    return ToolParam(
        name=_OUTPUT_TOOL_NAME,
        description="Emit the final answer for this request as structured data.",
        input_schema=output_schema.model_json_schema(),
    )


async def _call(
    client: AsyncAnthropic,
    *,
    model: str,
    system: str,
    messages: list[MessageParam],
    output_schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
) -> Message:
    return await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        tools=[_tool_param(output_schema)],
        tool_choice=ToolChoiceToolParam(type="tool", name=_OUTPUT_TOOL_NAME),
        output_config=OutputConfigParam(effort=_effort_for_temperature(temperature)),
    )


def _extract_tool_use_input(message: Message) -> dict[str, Any] | None:
    for block in message.content:
        if isinstance(block, ToolUseBlock) and block.name == _OUTPUT_TOOL_NAME:
            return block.input
    return None


async def get_structured_completion(
    client: AsyncAnthropic,
    *,
    model: str,
    system: str,
    messages: list[MessageParam],
    output_schema: type[T],
    prompt_version: str,
    temperature: float = 0.0,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> tuple[T, LLMCallMetadata]:
    """Forces `output_schema` as a single tool call, parses+validates the
    result with Pydantic, and retries exactly once on validation failure
    (guideline.md §7) with the validation error appended as an
    `is_error=True` tool_result turn. Raises `StructuredOutputError` if the
    model never produces the forced tool_use block, or if the second
    attempt also fails validation.
    """
    start = time.monotonic()
    total_input_tokens = 0
    total_output_tokens = 0
    retried = False

    response = await _call(
        client,
        model=model,
        system=system,
        messages=messages,
        output_schema=output_schema,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens

    parsed, error = _try_parse(response, output_schema)
    if parsed is None:
        retried = True
        tool_use_id = _tool_use_id(response)
        logger.info(
            "structured_output_retry",
            model=model,
            prompt_version=prompt_version,
            error=error,
        )
        retry_messages: list[MessageParam] = [
            *messages,
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id=tool_use_id,
                        content=f"Validation failed: {error}. Correct the input and call "
                        f"{_OUTPUT_TOOL_NAME} again with a schema-valid payload.",
                        is_error=True,
                    )
                ],
            },
        ]
        response = await _call(
            client,
            model=model,
            system=system,
            messages=retry_messages,
            output_schema=output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        parsed, error = _try_parse(response, output_schema)
        if parsed is None:
            raise StructuredOutputError(
                f"model never produced a schema-valid {output_schema.__name__} for "
                f"prompt_version={prompt_version!r} after 1 retry: {error}"
            )

    latency_ms = int((time.monotonic() - start) * 1000)
    metadata = LLMCallMetadata(
        model=model,
        prompt_version=prompt_version,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        latency_ms=latency_ms,
        cost_usd=estimate_cost_usd(
            model, input_tokens=total_input_tokens, output_tokens=total_output_tokens
        ),
        retried=retried,
    )
    return parsed, metadata


def _tool_use_id(response: Message) -> str:
    for block in response.content:
        if isinstance(block, ToolUseBlock) and block.name == _OUTPUT_TOOL_NAME:
            return block.id
    # Unreachable when `_try_parse` returned `None` for "no tool_use block
    # found" -- but if the model stopped for another reason (e.g. it argued
    # in plain text instead of calling the tool) there is no tool_use_id to
    # respond to; the retry turn still needs a well-formed transcript, so
    # fall back to a synthetic id the model has never seen. The provider
    # will reject a genuinely mismatched tool_use_id, which surfaces as a
    # second `StructuredOutputError` rather than a confusing lower-level one.
    return "missing_tool_use"


def _try_parse(response: Message, output_schema: type[T]) -> tuple[T | None, str | None]:
    raw_input = _extract_tool_use_input(response)
    if raw_input is None:
        return None, f"no {_OUTPUT_TOOL_NAME!r} tool_use block in the response"
    try:
        return output_schema.model_validate(raw_input), None
    except ValidationError as exc:
        return None, str(exc)
