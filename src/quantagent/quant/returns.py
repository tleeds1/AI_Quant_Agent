from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.quant.constants import TRADING_DAYS_PER_YEAR
from quantagent.quant.validation import assert_finite, assert_no_nan


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = P_t/P_{t-1} - 1 per column.

    The return convention used everywhere weights must be additive across
    assets (architecture.md §4.4) -- always use this, never `log_returns`,
    for portfolio aggregation. The first row is dropped (undefined).
    """
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """ln(P_t / P_{t-1}) per column.

    Only for explicit time-aggregation callers -- log returns are not
    weight-additive across assets and must never be used for portfolio
    aggregation (architecture.md §4.4).
    """
    ratio = prices / prices.shift(1)
    log_ratio = pd.DataFrame(np.log(ratio.to_numpy()), index=ratio.index, columns=ratio.columns)
    return log_ratio.iloc[1:]


def portfolio_returns(weights: pd.Series, asset_returns: pd.DataFrame) -> pd.Series:
    """r_p,t = sum_i w_i * r_i,t -- CURRENT weights applied to historical asset
    returns, never a historical NAV series (that embeds past weights,
    architecture.md §4.4).
    """
    if set(weights.index) != set(asset_returns.columns):
        raise ValueError("weights.index must match asset_returns.columns exactly")
    assert_no_nan(asset_returns, context="portfolio_returns")
    aligned_weights = weights.reindex(asset_returns.columns)
    return asset_returns.dot(aligned_weights)


def simulate_equity_curve(portfolio_returns: pd.Series, *, initial_value: float = 1.0) -> pd.Series:
    """V_t = initial_value * cumprod(1 + r_t). Feeds `drawdown.max_drawdown`."""
    assert_no_nan(portfolio_returns, context="simulate_equity_curve")
    return initial_value * (1.0 + portfolio_returns).cumprod()


def annualize_return(
    period_returns: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """(1 + mean(period_returns)) ** periods_per_year - 1."""
    value = float((1.0 + period_returns.mean()) ** periods_per_year - 1.0)
    assert_finite(value, context="annualize_return")
    return value


def annualize_volatility(
    period_returns: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """std(period_returns, ddof=1) * sqrt(periods_per_year)."""
    value = float(period_returns.std(ddof=1) * np.sqrt(periods_per_year))
    assert_finite(value, context="annualize_volatility")
    return value
