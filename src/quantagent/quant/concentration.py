from __future__ import annotations

import pandas as pd

from quantagent.quant.constants import TOP_N_HOLDINGS_CONCENTRATION
from quantagent.quant.types import ConcentrationResult
from quantagent.quant.validation import assert_finite, assert_no_nan, assert_weights_sum_to_one


def portfolio_concentration(
    weights: pd.Series, *, top_n: int = TOP_N_HOLDINGS_CONCENTRATION
) -> ConcentrationResult:
    """HHI = sum(w_i^2); effective_holdings = 1/HHI; top_n_weight = sum of the
    `top_n` largest positions by absolute weight (architecture.md §4.4). HHI
    is what turns "I have 30 stocks" into "you effectively have 6".

    NOTE: `HHI in [1/N, 1]` only holds for long-only weights (all w_i >= 0)
    -- a levered/short book can push HHI above 1. This function does not
    reject short weights (HHI is still well-defined), but the property test
    and this note document the bound is long-only.
    """
    assert_no_nan(weights, context="portfolio_concentration")
    assert_weights_sum_to_one(weights)

    hhi = float((weights**2).sum())
    assert_finite(hhi, context="portfolio_concentration.hhi")
    effective_holdings = 1.0 / hhi
    top_n_weight = float(weights.abs().nlargest(top_n).sum())

    return ConcentrationResult(
        method="hhi",
        sample_size=len(weights),
        hhi=hhi,
        effective_holdings=effective_holdings,
        top_n_weight=top_n_weight,
        top_n=top_n,
    )
