"""evals/eval_tool_selection.py -- architecture.md §10.3/§10.4's
tool-selection F1 gate (target >= 0.90), measured against real
`classify_intent`/`create_plan` calls -- the actual production INTAKE/PLAN
code path, not a re-implementation.

REQUIRES a live `ANTHROPIC_API_KEY`: tool selection is the planner's own
LLM judgment, so there is no deterministic way to measure it, and this
script does not fabricate a number when no key is configured -- it prints
what's missing and exits instead.

    uv run python -m evals.eval_tool_selection
"""

from __future__ import annotations

import asyncio

from evals.tool_selection_fixtures import GOLDEN_TRACES, GoldenTrace

from quantagent.agent.intent import classify_intent
from quantagent.agent.planner import create_plan
from quantagent.config import settings
from quantagent.contracts.errors import LLMError
from quantagent.llm.client import LLMClient
from quantagent.llm.prompts import PromptLoader


async def _tools_chosen_for(
    trace: GoldenTrace, *, client: LLMClient, prompts: PromptLoader
) -> set[str]:
    """Mirrors `agent/loop.py::_run_agent_loop_inner`'s own branching
    exactly: SIMPLE_LOOKUP resolves to its one `direct_tool`; everything
    else (PORTFOLIO_ANALYSIS/RESEARCH) goes through the full DAG planner;
    OUT_OF_SCOPE never reaches a tool at all.
    """
    intent = await classify_intent(
        trace.question, client=client, prompts=prompts, mandate_summary=None
    )
    if intent.label == "OUT_OF_SCOPE":
        return set()
    if intent.label == "SIMPLE_LOOKUP" and intent.direct_tool is not None:
        return {step.tool for step in intent.direct_tool.steps}
    plan, _llm_calls = await create_plan(trace.question, client=client, prompts=prompts)
    return {step.tool for step in plan.steps}


def _precision_recall_f1(actual: set[str], expected: frozenset[str]) -> tuple[float, float, float]:
    if not actual and not expected:
        return 1.0, 1.0, 1.0
    true_positive = len(actual & expected)
    precision = true_positive / len(actual) if actual else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


async def main() -> None:
    if not settings.anthropic_api_key.strip():
        print(
            "No ANTHROPIC_API_KEY configured -- tool-selection F1 requires a real planner call "
            "and this script will not fabricate a number without one.\n"
            "Set ANTHROPIC_API_KEY in .env, then re-run:\n"
            "    uv run python -m evals.eval_tool_selection"
        )
        return

    client = LLMClient(
        base_url=settings.anthropic_base_url,
        api_key=settings.anthropic_api_key,
    )
    prompts = PromptLoader()

    try:
        per_trace_f1: list[float] = []
        for trace in GOLDEN_TRACES:
            try:
                actual = await _tools_chosen_for(trace, client=client, prompts=prompts)
            except LLMError as exc:
                # A real LLM failure (unschema-valid output surviving its one
                # retry, or a transport error) is a genuine tool-selection
                # miss for this trace, not a bug in this harness -- record it
                # as F1=0 and keep measuring the rest, rather than losing the
                # whole run's signal to one bad trace (guideline.md §10.3's
                # gate needs an aggregate over all traces to mean anything).
                print(f"{trace.question!r}\n  FAILED: {exc}")
                per_trace_f1.append(0.0)
                continue
            precision, recall, f1 = _precision_recall_f1(actual, trace.expected_tools)
            per_trace_f1.append(f1)
            print(
                f"{trace.question!r}\n"
                f"  expected={sorted(trace.expected_tools)} actual={sorted(actual)}\n"
                f"  precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}"
            )
    finally:
        await client.close()

    aggregate_f1 = sum(per_trace_f1) / len(per_trace_f1)
    print(
        f"\n[tool-selection-f1] aggregate={aggregate_f1:.3f} over {len(GOLDEN_TRACES)} "
        f"golden traces (target >= 0.90, architecture.md §10.4)"
    )


if __name__ == "__main__":
    asyncio.run(main())
