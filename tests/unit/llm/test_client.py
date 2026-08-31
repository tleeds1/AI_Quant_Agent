from __future__ import annotations

import pytest
from pydantic import BaseModel

from quantagent.contracts.errors import StructuredOutputError
from quantagent.llm.client import _OUTPUT_TOOL_NAME, get_structured_completion
from tests.unit.llm.fixtures import build_mock_llm_client, text_only_response, tool_use_response


class _Choice(BaseModel):
    value: int


async def test_parses_forced_tool_use_response() -> None:
    client, session = build_mock_llm_client([tool_use_response(_OUTPUT_TOOL_NAME, {"value": 7})])

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


async def test_request_sends_system_as_first_message_and_real_temperature() -> None:
    client, session = build_mock_llm_client([tool_use_response(_OUTPUT_TOOL_NAME, {"value": 1})])

    await get_structured_completion(
        client,
        model="claude-haiku-4-5",
        system="pick a number",
        messages=[{"role": "user", "content": "go"}],
        output_schema=_Choice,
        prompt_version="test/x.v1",
        temperature=0.3,
    )

    body = session.request_body(0)
    assert body["temperature"] == 0.3
    assert body["messages"][0] == {"role": "system", "content": "pick a number"}
    assert body["tool_choice"] == {"type": "function", "function": {"name": _OUTPUT_TOOL_NAME}}


async def test_retries_once_on_validation_failure_then_succeeds() -> None:
    client, session = build_mock_llm_client(
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
    replayed_assistant_turn, retry_turn = second_body["messages"][-2:]
    assert replayed_assistant_turn["role"] == "assistant"
    assert replayed_assistant_turn["tool_calls"][0]["function"]["name"] == _OUTPUT_TOOL_NAME
    assert retry_turn["role"] == "tool"
    assert retry_turn["tool_call_id"] == "call_test"
    assert "Validation failed" in retry_turn["content"]


async def test_raises_structured_output_error_after_second_failure() -> None:
    client, _session = build_mock_llm_client(
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


async def test_recovers_json_written_as_fenced_text_instead_of_a_tool_call() -> None:
    """Some models don't reliably honor a forced tool_choice on a long
    generation and answer with the JSON directly as `content` -- observed
    live against gemma4:26b on the synthesis stage. Recovering it should
    succeed on the first attempt, no retry needed.
    """
    client, session = build_mock_llm_client([text_only_response('```json\n{"value": 9}\n```')])

    parsed, meta = await get_structured_completion(
        client,
        model="claude-haiku-4-5",
        system="pick a number",
        messages=[{"role": "user", "content": "go"}],
        output_schema=_Choice,
        prompt_version="test/x.v1",
    )

    assert parsed.value == 9
    assert meta.retried is False
    assert session.call_count == 1


async def test_missing_tool_use_block_raises_structured_output_error() -> None:
    client, _session = build_mock_llm_client([text_only_response("sorry, I refuse")])

    with pytest.raises(StructuredOutputError):
        await get_structured_completion(
            client,
            model="claude-haiku-4-5",
            system="pick a number",
            messages=[{"role": "user", "content": "go"}],
            output_schema=_Choice,
            prompt_version="test/x.v1",
        )
