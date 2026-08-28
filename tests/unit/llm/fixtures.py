"""tests/unit/llm/fixtures.py -- shared Anthropic-mocking helper for every
LLM-touching test in M3 (intent, planner, synthesizer, and the worked-example
e2e test).

`respx` cannot mock this SDK: `anthropic==1.0.0`'s httpx client is built on a
separate, real PyPI package literally named `httpx2` (its own dist-info,
version 2.12.0 -- not an alias of `httpx`; confirmed empirically that a
respx-wrapped call still hit the real network and got a real 401). `httpx2`
ships its own `MockTransport`, the same hand-rolled-transport pattern
`httpx.MockTransport` has long supported -- that is what this module wraps,
via the SDK's own supported `http_client=` injection point. See the project
memory note recorded alongside this milestone for the full finding.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx2
from anthropic import AsyncAnthropic

_DEFAULT_MODEL = "claude-haiku-4-5"


def tool_use_response(
    tool_name: str,
    input_: dict[str, Any],
    *,
    model: str = _DEFAULT_MODEL,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> dict[str, Any]:
    """One realistic Anthropic Messages API response envelope for a forced
    tool-use call -- the exact shape `llm/client.py::_try_parse` consumes.
    """
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "tool_use", "id": "toolu_test", "name": tool_name, "input": input_}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def text_only_response(text: str, *, model: str = _DEFAULT_MODEL) -> dict[str, Any]:
    """A response with no tool_use block -- exercises the "model refused to
    call the tool" failure path.
    """
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


class MockAnthropicSession:
    """Records every request and replays `responses` in order (the last
    response repeats if more calls happen than responses were supplied, so a
    test asserting on `requests`/`request_bodies` sees the extra call rather
    than an opaque `IndexError`).
    """

    def __init__(self, responses: Sequence[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        idx = min(len(self.requests) - 1, len(self._responses) - 1)
        return httpx2.Response(200, json=self._responses[idx])

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def request_body(self, index: int) -> dict[str, Any]:
        content = self.requests[index].content
        result: dict[str, Any] = json.loads(content)
        return result

    def build_client(self) -> AsyncAnthropic:
        transport = httpx2.MockTransport(self._handle)
        http_client = httpx2.AsyncClient(transport=transport)
        return AsyncAnthropic(api_key="test-key", http_client=http_client)


def build_mock_anthropic(
    responses: Sequence[dict[str, Any]],
) -> tuple[AsyncAnthropic, MockAnthropicSession]:
    session = MockAnthropicSession(responses)
    return session.build_client(), session
