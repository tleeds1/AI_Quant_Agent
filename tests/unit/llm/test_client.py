from __future__ import annotations

import pytest
from pydantic import BaseModel

from quantagent.contracts.errors import StructuredOutputError
from quantagent.llm.client import _OUTPUT_TOOL_NAME, get_structured_completion
from tests.unit.llm.fixtures import build_mock_anthropic, text_only_response, tool_use_response


class _Choice(BaseModel):
    value: int


async def test_parses_forced_tool_use_response() -> None:
    client, session = build_mock_anthropic([tool_use_response(_OUTPUT_TOOL_NAME, {"value": 7})])

    parsed, meta = await get_structured_completion(
        client,
        model="claude-haiku-4-5",
        system="pick a number",
        messages=[{"role": "user", "content": "go"}],
        output_schema=_Choice,
        prompt_version="test/x.v1",
    )

    assert parsed.value == 7
    assert meta.retried is False
    assert meta.input_tokens == 100
    assert meta.output_tokens == 20
    assert meta.prompt_version == "test/x.v1"
    assert meta.model == "claude-haiku-4-5"
    assert session.call_count == 1


async def test_request_uses_effort_not_temperature() -> None:
    client, session = build_mock_anthropic([tool_use_response(_OUTPUT_TOOL_NAME, {"value": 1})])

    await get_structured_completion(
        client,
        model="claude-haiku-4-5",
        system="s",
        messages=[{"role": "user", "content": "go"}],
        output_schema=_Choice,
        prompt_version="test/x.v1",
        temperature=0.0,
    )

    body = session.request_body(0)
    assert "temperature" not in body
    assert body["output_config"] == {"effort": "high"}


async def test_synthesis_temperature_maps_to_medium_effort() -> None:
    client, session = build_mock_anthropic([tool_use_response(_OUTPUT_TOOL_NAME, {"value": 1})])

    await get_structured_completion(
        client,
        model="claude-haiku-4-5",
        system="s",
        messages=[{"role": "user", "content": "go"}],
        output_schema=_Choice,
        prompt_version="test/x.v1",
        temperature=0.3,
    )

    assert session.request_body(0)["output_config"] == {"effort": "medium"}


async def test_retries_once_on_validation_failure_then_succeeds() -> None:
    client, session = build_mock_anthropic(
        [
            tool_use_response(_OUTPUT_TOOL_NAME, {"value": "not-an-int"}),
            tool_use_response(_OUTPUT_TOOL_NAME, {"value": 7}),
        ]
    )

    parsed, meta = await get_structured_completion(
        client,
        model="claude-haiku-4-5",
        system="pick a number",
        messages=[{"role": "user", "content": "go"}],
        output_schema=_Choice,
        prompt_version="test/x.v1",
    )

    assert parsed.value == 7
    assert meta.retried is True
    assert session.call_count == 2
    second_body = session.request_body(1)
    retry_turn = second_body["messages"][-1]["content"][0]
    assert retry_turn["type"] == "tool_result"
    assert retry_turn["is_error"] is True


async def test_raises_structured_output_error_after_second_failure() -> None:
    client, _session = build_mock_anthropic(
        [tool_use_response(_OUTPUT_TOOL_NAME, {"value": "nope"})]
    )

    with pytest.raises(StructuredOutputError):
        await get_structured_completion(
            client,
            model="claude-haiku-4-5",
            system="pick a number",
            messages=[{"role": "user", "content": "go"}],
            output_schema=_Choice,
            prompt_version="test/x.v1",
        )


async def test_missing_tool_use_block_raises_structured_output_error() -> None:
    client, _session = build_mock_anthropic([text_only_response("sorry, I refuse")])

    with pytest.raises(StructuredOutputError):
        await get_structured_completion(
            client,
            model="claude-haiku-4-5",
            system="pick a number",
            messages=[{"role": "user", "content": "go"}],
            output_schema=_Choice,
            prompt_version="test/x.v1",
        )
