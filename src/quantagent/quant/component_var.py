from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import MIN_CVAR_TAIL_OBSERVATIONS
from quantagent.quant.covariance import ledoit_wolf_covariance, portfolio_variance
from quantagent.quant.returns import portfolio_returns
from quantagent.quant.types import ComponentResult, CovarianceResult
from quantagent.quant.validation import assert_finite
from quantagent.quant.var import parametric_var


def parametric_component_var(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    alpha: float,
    *,
    cov: CovarianceResult | None = None,
) -> ComponentResult:
    """CVaR_i = w_i * (Sigma @ w)_i / Var_p * VaR_p (architecture.md §4.4).

    `Var_p = w^T Sigma w` is portfolio VARIANCE, not volatility: dividing by
    variance (not `sigma_p = sqrt(Var_p)`) is what makes the sum-to-VaR_p
    identity exact -- `w_i * (Sigma w)_i / Var_p` is asset i's beta to the
    portfolio (`Cov(r_i, r_p) / Var(r_p)`), and betas weighted by `w_i` sum
    to 1 by construction. `sum(components.values()) == portfolio VaR`
    (property-tested in tests/property/test_component_var_properties.py) --
    this is the metric that answers "am I overexposed?": a 22% weight
    driving 48% of tail risk is the finding, not the weight.
    """
    covariance = cov if cov is not None else ledoit_wolf_covariance(asset_returns)
    aligned_weights = weights.reindex(covariance.matrix.index)
    portfolio_var_value = portfolio_variance(weights, covariance.matrix)
    var_p = parametric_var(weights, asset_returns, alpha, cov=covariance)

    sigma_w = covariance.matrix.to_numpy(dtype=np.float64) @ aligned_weights.to_numpy(
        dtype=np.float64
    )
    raw_components = (
        aligned_weights.to_numpy(dtype=np.float64) * sigma_w / portfolio_var_value * var_p.value
    )
    components = dict(zip(aligned_weights.index, raw_components.tolist(), strict=True))
    for ticker, value in components.items():
        assert_finite(value, context=f"parametric_component_var[{ticker}]")

    return ComponentResult(
        method="parametric",
        sample_size=covariance.sample_size,
        components=components,
        portfolio_value=var_p.value,
    )


def historical_component_var(
    weights: pd.Series, asset_returns: pd.DataFrame, alpha: float
) -> ComponentResult:
    """Per-asset contribution = -w_i * mean(r_i,t | portfolio in its own
    historical VaR tail) (architecture.md §4.4).

    NOTE: unlike the parametric case, these contributions do NOT sum exactly
    to portfolio VaR -- summing `-w_i * mean(r_i | tail)` over i recovers
    `-mean(r_p | tail)`, which is CVaR's definition, not VaR's. The exact
    sum-to-portfolio-VaR identity only holds for `parametric_component_var`.
    """
    r_p = portfolio_returns(weights, asset_returns)
    threshold = np.quantile(r_p.to_numpy(), 1.0 - alpha)
    tail_mask = r_p <= threshold
    if int(tail_mask.sum()) < MIN_CVAR_TAIL_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_CVAR_TAIL_OBSERVATIONS} tail observations for component "
            f"VaR, got {int(tail_mask.sum())}"
        )

    aligned_weights = weights.reindex(asset_returns.columns)
    tail_means = asset_returns.loc[tail_mask].mean()
    raw_components = -(aligned_weights * tail_means)
    components = {str(ticker): float(value) for ticker, value in raw_components.items()}
    for ticker, value in components.items():
        assert_finite(value, context=f"historical_component_var[{ticker}]")

    var_p = float(-threshold)
    return ComponentResult(
        method="historical",
        sample_size=len(r_p),
        components=components,
        portfolio_value=var_p,
    )
