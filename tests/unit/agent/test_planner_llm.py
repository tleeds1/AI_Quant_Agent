from __future__ import annotations

import pytest

from quantagent.agent.planner import PROMPT_STAGE, create_plan
from quantagent.contracts.errors import ToolValidationError
from quantagent.llm.prompts import PromptLoader
from quantagent.tools.registry import registry
from tests.unit.llm.fixtures import build_mock_llm_client, tool_use_response

_OUTPUT_TOOL = "emit_structured_output"


def _valid_plan_payload() -> dict[str, object]:
    return {
        "steps": [
            {
                "id": "s1",
                "tool": "get_holdings",
                "args": {"portfolio_id": "pf_1"},
                "depends_on": [],
            },
            {
                "id": "s2",
                "tool": "calculate_portfolio_var",
                "args": {"portfolio_id": "pf_1"},
                "depends_on": ["s1"],
            },
        ],
        "success_criteria": "assess VaR",
    }


async def test_create_plan_happy_path_single_call() -> None:
    client, session = build_mock_llm_client(
        [tool_use_response(_OUTPUT_TOOL, _valid_plan_payload())]
    )
    prompts = PromptLoader()

    plan, metas = await create_plan(
        "what's my portfolio VaR?", client=client, prompts=prompts, registry=registry
    )

    assert [s.tool for s in plan.steps] == ["get_holdings", "calculate_portfolio_var"]
    assert len(metas) == 1
    assert session.call_count == 1
    assert metas[0].prompt_version == f"{PROMPT_STAGE}/dag.v1"


async def test_create_plan_reprompts_once_on_semantically_invalid_plan() -> None:
    invalid_payload = {
        "steps": [{"id": "s1", "tool": "not_a_real_tool", "args": {}, "depends_on": []}],
        "success_criteria": "x",
    }
    client, session = build_mock_llm_client(
        [
            tool_use_response(_OUTPUT_TOOL, invalid_payload),
            tool_use_response(_OUTPUT_TOOL, _valid_plan_payload()),
        ]
    )
    prompts = PromptLoader()

    plan, metas = await create_plan(
        "what's my VaR?", client=client, prompts=prompts, registry=registry
    )

    assert [s.tool for s in plan.steps] == ["get_holdings", "calculate_portfolio_var"]
    assert len(metas) == 2
    assert session.call_count == 2
    assert metas[1].prompt_version == f"{PROMPT_STAGE}/dag_repair.v1"


async def test_create_plan_raises_after_repair_still_invalid() -> None:
    invalid_payload = {
        "steps": [{"id": "s1", "tool": "not_a_real_tool", "args": {}, "depends_on": []}],
        "success_criteria": "x",
    }
    client, session = build_mock_llm_client(
        [
            tool_use_response(_OUTPUT_TOOL, invalid_payload),
            tool_use_response(_OUTPUT_TOOL, invalid_payload),
        ]
    )
    prompts = PromptLoader()

    with pytest.raises(ToolValidationError):
        await create_plan("what's my VaR?", client=client, prompts=prompts, registry=registry)

    assert session.call_count == 2
