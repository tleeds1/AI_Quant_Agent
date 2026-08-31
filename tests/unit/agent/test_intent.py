from __future__ import annotations

from quantagent.agent.intent import classify_intent
from quantagent.llm.prompts import PromptLoader
from quantagent.tools.registry import registry
from tests.unit.llm.fixtures import build_mock_llm_client, tool_use_response

_OUTPUT_TOOL = "emit_structured_output"


async def test_out_of_scope_has_no_direct_tool() -> None:
    payload = {"label": "OUT_OF_SCOPE", "confidence": 0.95, "rationale": "not finance-related"}
    client, _session = build_mock_llm_client([tool_use_response(_OUTPUT_TOOL, payload)])

    result = await classify_intent(
        "what's the weather today?", client=client, prompts=PromptLoader(), registry=registry
    )

    assert result.label == "OUT_OF_SCOPE"
    assert result.direct_tool is None


async def test_portfolio_analysis_has_no_direct_tool() -> None:
    payload = {
        "label": "PORTFOLIO_ANALYSIS",
        "confidence": 0.8,
        "rationale": "needs multi-step analysis",
    }
    client, _session = build_mock_llm_client([tool_use_response(_OUTPUT_TOOL, payload)])

    result = await classify_intent(
        "why did my drawdown spike last month?",
        client=client,
        prompts=PromptLoader(),
        registry=registry,
    )

    assert result.label == "PORTFOLIO_ANALYSIS"
    assert result.direct_tool is None


async def test_research_has_no_direct_tool() -> None:
    payload = {
        "label": "RESEARCH",
        "confidence": 0.85,
        "rationale": "company-research question, no portfolio data needed",
    }
    client, _session = build_mock_llm_client([tool_use_response(_OUTPUT_TOOL, payload)])

    result = await classify_intent(
        "what does NVDA's 10-K say about supply-chain risk?",
        client=client,
        prompts=PromptLoader(),
        registry=registry,
    )

    assert result.label == "RESEARCH"
    assert result.direct_tool is None


async def test_simple_lookup_with_valid_tool_resolves_to_one_step_plan() -> None:
    payload = {
        "label": "SIMPLE_LOOKUP",
        "confidence": 0.9,
        "rationale": "single tool answers this",
        "direct_tool": {"tool_name": "get_holdings", "args": {"portfolio_id": "pf_1"}},
    }
    client, session = build_mock_llm_client([tool_use_response(_OUTPUT_TOOL, payload)])

    result = await classify_intent(
        "what are my current holdings?", client=client, prompts=PromptLoader(), registry=registry
    )

    assert result.label == "SIMPLE_LOOKUP"
    assert result.direct_tool is not None
    assert len(result.direct_tool.steps) == 1
    assert result.direct_tool.steps[0].tool == "get_holdings"
    assert session.call_count == 1


async def test_simple_lookup_with_invalid_args_downgrades_to_portfolio_analysis() -> None:
    # A registered tool (calculate_portfolio_var) but args failing its own
    # schema (alpha must be in [0.90, 0.999]) -- validate_plan inside
    # `_finalize` catches this; it must downgrade rather than error, and
    # must NOT trigger a second LLM round trip (the full planner is the
    # safety net for a bad direct-tool choice, not a bespoke retry here).
    invalid_args_payload = {
        "label": "SIMPLE_LOOKUP",
        "confidence": 0.9,
        "rationale": "x",
        "direct_tool": {
            "tool_name": "calculate_portfolio_var",
            "args": {"portfolio_id": "pf_1", "alpha": 5.0},
        },
    }
    client, session = build_mock_llm_client([tool_use_response(_OUTPUT_TOOL, invalid_args_payload)])

    result = await classify_intent(
        "what's my VaR?", client=client, prompts=PromptLoader(), registry=registry
    )

    assert result.label == "PORTFOLIO_ANALYSIS"
    assert result.direct_tool is None
    assert "downgraded" in result.rationale
    assert session.call_count == 1


async def test_mandate_summary_none_renders_without_error() -> None:
    payload = {"label": "OUT_OF_SCOPE", "confidence": 0.5, "rationale": "x"}
    client, _session = build_mock_llm_client([tool_use_response(_OUTPUT_TOOL, payload)])

    result = await classify_intent(
        "hello", client=client, prompts=PromptLoader(), mandate_summary=None, registry=registry
    )

    assert result.label == "OUT_OF_SCOPE"
