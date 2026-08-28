"""llm/pricing.py -- cost estimation for LLM calls (architecture.md §9.1's
"all LLM calls: prompt hash, model, tokens, cost, latency" trace requirement).

Prices are USD per 1M tokens, input/output. An unpriced model returns 0.0 and
logs a warning rather than raising -- a missing price entry must never abort
a request, but must also never be silently invisible in the trace.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

# Deliberately narrow: only the models this project's config.py actually
# references (settings.model_intent/planner/synthesizer/critic). Extend when
# a new model is configured, not speculatively.
_PRICE_PER_1M_TOKENS_USD: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}


def estimate_cost_usd(model: str, *, input_tokens: int, output_tokens: int) -> float:
    prices = _PRICE_PER_1M_TOKENS_USD.get(model)
    if prices is None:
        logger.warning("llm_pricing_unknown_model", model=model)
        return 0.0
    input_price, output_price = prices
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
