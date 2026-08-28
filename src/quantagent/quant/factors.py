from __future__ import annotations

import math

import pandas as pd
import statsmodels.api as sm

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import (
    FACTOR_TSTAT_SIGNIFICANCE_THRESHOLD,
    MIN_FACTOR_REGRESSION_OBSERVATIONS,
)
from quantagent.quant.types import FactorExposureResult
from quantagent.quant.validation import assert_aligned_index, assert_finite, assert_no_nan


def factor_exposure(
    portfolio_excess_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    hac_lags: int | None = None,
) -> FactorExposureResult:
    """OLS of portfolio excess returns on `factor_returns`' columns (caller
    passes FF5+MOM columns per architecture.md §4.4 -- this function is
    factor-set-agnostic by design), with HAC (Newey-West) standard errors.
    `hac_lags` defaults to the Newey-West (1994) automatic rule
    `floor(4*(T/100)**(2/9))` when `None`. Reports per-factor beta, t-stat,
    a `significant` flag (`|t| >= FACTOR_TSTAT_SIGNIFICANCE_THRESHOLD`), R²,
    and idiosyncratic (residual) variance share. Raises
    `InsufficientDataError` below `MIN_FACTOR_REGRESSION_OBSERVATIONS` or
    when the regression is underdetermined (T < 10 * factor count).
    """
    assert_no_nan(portfolio_excess_returns, context="factor_exposure")
    assert_no_nan(factor_returns, context="factor_exposure")
    assert_aligned_index(portfolio_excess_returns, factor_returns, context="factor_exposure")

    n_obs = len(portfolio_excess_returns)
    n_factors = factor_returns.shape[1]
    if n_obs < MIN_FACTOR_REGRESSION_OBSERVATIONS or n_obs < 10 * n_factors:
        raise InsufficientDataError(
            f"need at least max({MIN_FACTOR_REGRESSION_OBSERVATIONS}, {10 * n_factors}) "
            f"observations for a {n_factors}-factor regression, got {n_obs}"
        )

    lags = hac_lags if hac_lags is not None else math.floor(4 * (n_obs / 100) ** (2 / 9))
    design_matrix = sm.add_constant(factor_returns)
    fitted = sm.OLS(portfolio_excess_returns, design_matrix).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )

    factor_names = list(factor_returns.columns)
    betas = {name: float(fitted.params[name]) for name in factor_names}
    t_stats = {name: float(fitted.tvalues[name]) for name in factor_names}
    significant = {
        name: abs(t_stats[name]) >= FACTOR_TSTAT_SIGNIFICANCE_THRESHOLD for name in factor_names
    }
    r_squared = float(fitted.rsquared)
    idiosyncratic_share = 1.0 - r_squared
    assert_finite(r_squared, context="factor_exposure.r_squared")

    return FactorExposureResult(
        method="ols_hac",
        sample_size=n_obs,
        betas=betas,
        t_stats=t_stats,
        significant=significant,
        r_squared=r_squared,
        idiosyncratic_variance_share=idiosyncratic_share,
        hac_lags=lags,
    )
