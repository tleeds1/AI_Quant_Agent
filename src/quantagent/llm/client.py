"""llm/client.py -- the one shared structured-output primitive every LLM-
calling stage (INTAKE, PLAN, SYNTHESIZE) builds on (guideline.md §7:
"Structured output via tool/function schema, not 'respond in JSON'. Parse
with Pydantic and retry once on validation failure with the validation error
appended.").

Talks to an OpenAI-Chat-Completions-compatible `/chat/completions` endpoint
via plain httpx, not a vendor SDK -- this project points it at Open WebUI's
proxy in front of the company's local models (config.py's
`anthropic_base_url`/`anthropic_api_key`, see .env.example). Forces the
target schema as a single named tool call via `tools`/`tool_choice`,
matching guideline.md §7's structured-output policy without depending on any
one vendor's tool-calling wire format. `temperature` is sent as-is: unlike
Anthropic's Messages API (which has no `temperature` parameter, only a
reasoning-effort control), the OpenAI-compatible surface accepts it
directly, so no effort-mapping workaround is needed here.

Smaller/local models don't always honor a forced `tool_choice` on a long
generation -- observed live against `gemma4:26b`: the synthesis stage (a
large prompt, a large output schema) came back with the answer as a
```json ... ``` fenced code block in plain `content`, no `tool_calls` at
all. `_extract_json_from_text` recovers that case rather than spend the one
permitted retry on a request the model is just as likely to answer the same
way again.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import structlog
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from quantagent.contracts.errors import LLMTransportError, StructuredOutputError
from quantagent.llm.pricing import estimate_cost_usd

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

MessageParam = dict[str, Any]

_OUTPUT_TOOL_NAME = "emit_structured_output"
_DEFAULT_MAX_TOKENS = 2048
_HTTP_TIMEOUT_S = 60.0
_MAX_ATTEMPTS = 3


def _is_transient_http_error(exc: BaseException) -> bool:
    """Only rate-limit (429) and server errors (5xx) are retried, mirroring
    `data/providers/edgar.py`'s policy -- a 4xx other than 429 (bad request,
    unauthorized, model not found) is permanent and retrying it would waste
    the request budget on a call that can never succeed.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class LLMClient:
    """Thin async wrapper around an OpenAI-Chat-Completions-compatible
    endpoint. `base_url` is the API root without a trailing
    `/chat/completions` (this class appends it) -- e.g. Open WebUI's
    `<host>/api`. Holds one long-lived `httpx.AsyncClient` for connection
    reuse across the many calls a single request's INTAKE/PLAN/SYNTHESIZE
    stages make, matching how the SDK client it replaces behaved. No I/O
    happens at construction, only at the first call.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # `transport` is test-only (tests inject an `httpx.MockTransport`);
        # `None` is httpx's own default and constructs the real transport.
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_HTTP_TIMEOUT_S,
            transport=transport,
        )

    async def close(self) -> None:
        await self._http.aclose()

    @retry(
        retry=retry_if_exception(_is_transient_http_error),
        wait=wait_exponential(multiplier=0.5, max=5),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        reraise=True,
    )
    async def post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        return dict(response.json())


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


@dataclass(frozen=True, slots=True)
class _ToolCallResult:
    tool_input: dict[str, Any] | None
    tool_call_id: str | None
    raw_assistant_message: dict[str, Any]
    input_tokens: int
    output_tokens: int


def _tool_schema(output_schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _OUTPUT_TOOL_NAME,
            "description": "Emit the final answer for this request as structured data.",
            "parameters": output_schema.model_json_schema(),
        },
    }


_JSON_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_json_from_text(content: str) -> dict[str, Any] | None:
    """Recovers a JSON object a model wrote as plain `content` instead of a
    forced tool call (see module docstring) -- a fenced ```json ... ``` block
    if present, otherwise the whole trimmed string. `None` if nothing in it
    parses as a JSON object, so the caller falls through to its normal
    "no tool call" failure path unchanged.
    """
    if not content:
        return None
    match = _JSON_CODE_FENCE_RE.search(content)
    candidate = match.group(1) if match else content.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_response(payload: dict[str, Any]) -> _ToolCallResult:
    choices = payload.get("choices") or []
    message: dict[str, Any] = choices[0]["message"] if choices else {}

    tool_input: dict[str, Any] | None = None
    tool_call_id: str | None = None
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        if function.get("name") != _OUTPUT_TOOL_NAME:
            continue
        tool_call_id = call.get("id")
        try:
            tool_input = json.loads(function.get("arguments") or "")
        except json.JSONDecodeError:
            tool_input = None
        break

    if tool_input is None:
        tool_input = _extract_json_from_text(message.get("content") or "")

    usage = payload.get("usage") or {}
    return _ToolCallResult(
        tool_input=tool_input,
        tool_call_id=tool_call_id,
        raw_assistant_message=message,
        input_tokens=int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        output_tokens=int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
    )


async def _call(
    client: LLMClient,
    *,
    model: str,
    system: str,
    messages: list[MessageParam],
    output_schema: type[BaseModel],
    temperature: float,
    max_tokens: int,
) -> _ToolCallResult:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": [_tool_schema(output_schema)],
        "tool_choice": {"type": "function", "function": {"name": _OUTPUT_TOOL_NAME}},
    }
    try:
        raw = await client.post_chat_completion(payload)
    except httpx.HTTPError as exc:
        raise LLMTransportError(
            f"chat completion request failed for model={model!r}: {exc}"
        ) from exc
    return _parse_response(raw)


async def get_structured_completion(
    client: LLMClient,
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
    (guideline.md §7) with the validation error appended as a `role="tool"`
    turn. Raises `StructuredOutputError` if the model never produces the
    forced tool call, or if the second attempt also fails validation.
    """
    start = time.monotonic()
    total_input_tokens = 0
    total_output_tokens = 0
    retried = False

    result = await _call(
        client,
        model=model,
        system=system,
        messages=messages,
        output_schema=output_schema,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    total_input_tokens += result.input_tokens
    total_output_tokens += result.output_tokens

    parsed, error = _try_parse(result, output_schema)
    if parsed is None:
        retried = True
        logger.info(
            "structured_output_retry",
            model=model,
            prompt_version=prompt_version,
            error=error,
        )
        retry_messages: list[MessageParam] = [
            *messages,
            result.raw_assistant_message,
            {
                "role": "tool",
                "tool_call_id": result.tool_call_id or "missing_tool_call",
                "content": f"Validation failed: {error}. Correct the input and call "
                f"{_OUTPUT_TOOL_NAME} again with a schema-valid payload.",
            },
        ]
        result = await _call(
            client,
            model=model,
            system=system,
            messages=retry_messages,
            output_schema=output_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        parsed, error = _try_parse(result, output_schema)
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


def _try_parse(result: _ToolCallResult, output_schema: type[T]) -> tuple[T | None, str | None]:
    if result.tool_input is None:
        return None, f"no {_OUTPUT_TOOL_NAME!r} tool call in the response"
    try:
        return output_schema.model_validate(result.tool_input), None
    except ValidationError as exc:
        return None, str(exc)
