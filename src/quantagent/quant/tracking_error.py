from __future__ import annotations

import pandas as pd

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import MIN_TRACKING_ERROR_OBSERVATIONS
from quantagent.quant.returns import annualize_volatility, portfolio_returns
from quantagent.quant.types import ScalarResult
from quantagent.quant.validation import assert_aligned_index


def tracking_error(
    weights: pd.Series, asset_returns: pd.DataFrame, benchmark_returns: pd.Series
) -> ScalarResult:
    """TE = std(r_p - r_b, ddof=1) * sqrt(TRADING_DAYS_PER_YEAR): the annualised
    standard deviation of active (portfolio minus benchmark) daily returns.

    Sign convention: always non-negative -- this is a dispersion measure, so
    0.04 reads as "4% annualised tracking error" and never carries a
    direction. Raises `ValueError` if `benchmark_returns`' index doesn't
    match the portfolio returns' index (pre-align upstream via
    `calendar.align_calendars`). Raises `InsufficientDataError` below
    `MIN_TRACKING_ERROR_OBSERVATIONS`.
    """
    r_p = portfolio_returns(weights, asset_returns)
    assert_aligned_index(r_p, benchmark_returns, context="tracking_error")
    if len(r_p) < MIN_TRACKING_ERROR_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_TRACKING_ERROR_OBSERVATIONS} observations for tracking "
            f"error, got {len(r_p)}"
        )
    value = annualize_volatility(r_p - benchmark_returns)
    return ScalarResult(method="annualised_tracking_error", sample_size=len(r_p), value=value)
