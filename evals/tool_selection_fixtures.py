"""evals/tool_selection_fixtures.py -- architecture.md §10.3's "golden
traces": expected tool set per question (order/DAG-dependency-agnostic,
matching the architecture doc's own "expected ordering constraints, not
exact sequence" framing). A small, hand-curated set (8, not the ~60
architecture.md sketches) -- enough to compute a real number, not a claim
that this is the full production eval set.

Measuring against these needs a live `ANTHROPIC_API_KEY` (tool selection is
the planner's own LLM judgment) -- see `evals/eval_tool_selection.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

DEMO_PORTFOLIO_ID = "pf_demo"


@dataclass(frozen=True, slots=True)
class GoldenTrace:
    question: str
    portfolio_id: str | None
    expected_tools: frozenset[str]


GOLDEN_TRACES: list[GoldenTrace] = [
    GoldenTrace(
        "What is my portfolio's benchmark?", DEMO_PORTFOLIO_ID, frozenset({"get_portfolio"})
    ),
    GoldenTrace("What are my current holdings?", DEMO_PORTFOLIO_ID, frozenset({"get_holdings"})),
    GoldenTrace(
        "What is my portfolio's 1-day 95% VaR?",
        DEMO_PORTFOLIO_ID,
        frozenset({"calculate_portfolio_var"}),
    ),
    GoldenTrace(
        "How concentrated is my portfolio? Show me HHI and my top holdings.",
        DEMO_PORTFOLIO_ID,
        frozenset({"get_concentration_metrics"}),
    ),
    GoldenTrace(
        "What is my portfolio's beta versus the market, and how correlated are my holdings?",
        DEMO_PORTFOLIO_ID,
        frozenset({"get_portfolio_beta", "get_correlation_matrix"}),
    ),
    GoldenTrace(
        "What does NVDA's 10-K say about supply-chain risk?",
        None,
        frozenset({"retrieve_company_filings"}),
    ),
    GoldenTrace(
        "Am I overexposed to a single sector, and which holdings drive my portfolio's tail risk?",
        DEMO_PORTFOLIO_ID,
        frozenset({"get_sector_exposure", "calculate_component_var"}),
    ),
    GoldenTrace(
        "What is NVDA's latest revenue and P/E ratio?", None, frozenset({"get_fundamentals"})
    ),
]
