"""tests/unit/llm/fixtures.py -- shared LLM-mocking helper for every
LLM-touching test (intent, planner, synthesizer, verifier, and the
worked-example e2e test).

`LLMClient` talks plain `httpx` to an OpenAI-Chat-Completions-compatible
`/chat/completions` endpoint, so a plain `httpx.MockTransport` injected via
`LLMClient`'s own `transport=` constructor arg is enough -- no global
monkeypatching (respx's default `@respx.mock` router) needed, and no
lifecycle to remember to tear down between tests.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx

from quantagent.llm.client import LLMClient

_DEFAULT_MODEL = "claude-haiku-4-5"
_TEST_BASE_URL = "https://llm.test/api"


def tool_use_response(
    tool_name: str,
    input_: dict[str, Any],
    *,
    model: str = _DEFAULT_MODEL,
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> dict[str, Any]:
    """One realistic OpenAI-Chat-Completions response envelope for a forced
    tool-call -- the exact shape `llm/client.py::_parse_response` consumes.
    """
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_test",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(input_)},
                        }
                    ],
                },
            }
        ],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def text_only_response(text: str, *, model: str = _DEFAULT_MODEL) -> dict[str, Any]:
    """A response with no tool call -- exercises the "model refused to call
    the tool" failure path.
    """
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {"input_tokens": 50, "output_tokens": 10},
    }


class MockLLMSession:
    """Records every request and replays `responses` in order (the last
    response repeats if more calls happen than responses were supplied, so a
    test asserting on `requests`/`request_body` sees the extra call rather
    than an opaque `IndexError`).
    """

    def __init__(self, responses: Sequence[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content)
        self.requests.append(body)
        idx = min(len(self.requests) - 1, len(self._responses) - 1)
        return httpx.Response(200, json=self._responses[idx])

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def request_body(self, index: int) -> dict[str, Any]:
        return self.requests[index]

    def build_client(self) -> LLMClient:
        transport = httpx.MockTransport(self._handle)
        return LLMClient(base_url=_TEST_BASE_URL, api_key="test-key", transport=transport)


def build_mock_llm_client(
    responses: Sequence[dict[str, Any]],
) -> tuple[LLMClient, MockLLMSession]:
    session = MockLLMSession(responses)
    return session.build_client(), session
