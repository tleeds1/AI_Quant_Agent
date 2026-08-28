from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import MIN_BETA_OBSERVATIONS
from quantagent.quant.returns import portfolio_returns
from quantagent.quant.types import ScalarResult
from quantagent.quant.validation import assert_aligned_index, assert_finite


def beta(
    weights: pd.Series, asset_returns: pd.DataFrame, market_returns: pd.Series
) -> ScalarResult:
    """beta = Cov(r_p, r_m) / Var(r_m) (architecture.md §4.4).

    Raises `ValueError` if `market_returns`' index doesn't match the
    portfolio returns' index (must be pre-aligned upstream via
    `calendar.align_calendars`). Raises `InsufficientDataError` below
    `MIN_BETA_OBSERVATIONS`.
    """
    r_p = portfolio_returns(weights, asset_returns)
    assert_aligned_index(r_p, market_returns, context="beta")
    if len(r_p) < MIN_BETA_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_BETA_OBSERVATIONS} observations for beta, got {len(r_p)}"
        )
    value = _covariance_over_variance(r_p, market_returns)
    assert_finite(value, context="beta")
    return ScalarResult(method="ols_beta", sample_size=len(r_p), value=value)


def downside_beta(
    weights: pd.Series, asset_returns: pd.DataFrame, market_returns: pd.Series
) -> ScalarResult:
    """Same formula as `beta`, restricted to days where `market_returns < 0` --
    the asymmetry is usually the interesting part (architecture.md §4.4).
    """
    r_p = portfolio_returns(weights, asset_returns)
    assert_aligned_index(r_p, market_returns, context="downside_beta")
    down_mask = market_returns < 0
    r_p_down = r_p[down_mask]
    r_m_down = market_returns[down_mask]
    if len(r_p_down) < MIN_BETA_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_BETA_OBSERVATIONS} down-market observations for "
            f"downside beta, got {len(r_p_down)}"
        )
    value = _covariance_over_variance(r_p_down, r_m_down)
    assert_finite(value, context="downside_beta")
    return ScalarResult(method="ols_downside_beta", sample_size=len(r_p_down), value=value)


def _covariance_over_variance(r_p: pd.Series, r_m: pd.Series) -> float:
    covariance_matrix = np.cov(r_p.to_numpy(dtype=np.float64), r_m.to_numpy(dtype=np.float64))
    return float(covariance_matrix[0, 1] / covariance_matrix[1, 1])
