from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import (
    HORIZON_SCALING_MAX_RELIABLE_DAYS,
    MIN_VAR_OBSERVATIONS,
    MONTE_CARLO_DEFAULT_SIMULATIONS,
    MONTE_CARLO_MIN_SIMULATIONS,
    MONTE_CARLO_T_DIST_DEFAULT_DOF,
)
from quantagent.quant.covariance import ledoit_wolf_covariance, portfolio_variance
from quantagent.quant.returns import portfolio_returns
from quantagent.quant.types import CovarianceResult, TailRiskResult
from quantagent.quant.validation import assert_finite

Method = Literal["historical", "parametric", "monte_carlo"]


def historical_var(weights: pd.Series, asset_returns: pd.DataFrame, alpha: float) -> TailRiskResult:
    """VaR_alpha = -quantile(r_p, 1-alpha) on the empirical distribution of
    portfolio returns from CURRENT weights applied to historical asset
    returns (architecture.md §4.4). Positive loss fraction.
    """
    r_p = portfolio_returns(weights, asset_returns)
    if len(r_p) < MIN_VAR_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_VAR_OBSERVATIONS} observations for historical VaR, "
            f"got {len(r_p)}"
        )
    value = float(-np.quantile(r_p.to_numpy(), 1.0 - alpha))
    assert_finite(value, context="historical_var")
    return TailRiskResult(method="historical", sample_size=len(r_p), value=value, alpha=alpha)


def parametric_var(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    alpha: float,
    *,
    cov: CovarianceResult | None = None,
) -> TailRiskResult:
    """VaR_alpha = -(mu_p + z_(1-alpha) * sigma_p), z = scipy.stats.norm.ppf(1-alpha)
    (architecture.md §4.4). Always carries a normality caveat: equity
    portfolios are fat-tailed and this understates the tail.
    """
    r_p = portfolio_returns(weights, asset_returns)
    covariance = cov if cov is not None else ledoit_wolf_covariance(asset_returns)
    mu_p = float(r_p.mean())
    sigma_p = float(np.sqrt(portfolio_variance(weights, covariance.matrix)))
    z = float(stats.norm.ppf(1.0 - alpha))
    value = -(mu_p + z * sigma_p)
    assert_finite(value, context="parametric_var")
    return TailRiskResult(
        method="parametric",
        sample_size=covariance.sample_size,
        value=value,
        alpha=alpha,
        warnings=[
            "parametric VaR assumes normality; understates tail risk for fat-tailed equity returns"
        ],
    )


def monte_carlo_var(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    alpha: float,
    *,
    n_sims: int = MONTE_CARLO_DEFAULT_SIMULATIONS,
    innovation: Literal["t", "bootstrap"] = "t",
    t_dof: float = MONTE_CARLO_T_DIST_DEFAULT_DOF,
    seed: int | None = None,
) -> TailRiskResult:
    """Simulates `n_sims` 1-day portfolio return draws from the Ledoit-Wolf-shrunk
    covariance. 't' innovations preserve cross-asset correlation via a
    normal-variance-mixture construction; 'bootstrap' resamples historical
    return rows with replacement. VaR = -quantile(simulated r_p, 1-alpha)
    (architecture.md §4.4). The seed actually used is always echoed back
    (guideline.md §6 rule 10), generated fresh when the caller passes None.
    """
    if n_sims < MONTE_CARLO_MIN_SIMULATIONS:
        raise ValueError(f"n_sims must be >= {MONTE_CARLO_MIN_SIMULATIONS}, got {n_sims}")
    covariance = ledoit_wolf_covariance(asset_returns)
    used_seed = seed if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    rng = np.random.default_rng(used_seed)
    aligned_weights = weights.reindex(asset_returns.columns).to_numpy(dtype=np.float64)

    simulated_asset_returns = _simulate_asset_returns(
        asset_returns, covariance, rng, n_sims, innovation, t_dof
    )
    simulated_portfolio_returns = simulated_asset_returns @ aligned_weights

    value = float(-np.quantile(simulated_portfolio_returns, 1.0 - alpha))
    assert_finite(value, context="monte_carlo_var")
    return TailRiskResult(
        method=f"monte_carlo_{innovation}",
        sample_size=n_sims,
        value=value,
        alpha=alpha,
        seed=used_seed,
    )


def _simulate_asset_returns(
    asset_returns: pd.DataFrame,
    covariance: CovarianceResult,
    rng: np.random.Generator,
    n_sims: int,
    innovation: Literal["t", "bootstrap"],
    t_dof: float,
) -> NDArray[np.float64]:
    """Draw `n_sims` simulated multi-asset daily return rows."""
    if innovation == "bootstrap":
        row_indices = rng.integers(0, len(asset_returns), size=n_sims)
        return asset_returns.to_numpy(dtype=np.float64)[row_indices]

    mu = asset_returns.mean().to_numpy(dtype=np.float64)
    cholesky_factor = np.linalg.cholesky(covariance.matrix.to_numpy(dtype=np.float64))
    normal_draws = rng.standard_normal((n_sims, len(mu)))
    chi_square_mix = rng.chisquare(t_dof, size=n_sims) / t_dof
    return mu + (normal_draws @ cholesky_factor.T) / np.sqrt(chi_square_mix)[:, None]


def portfolio_var(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    alpha: float,
    *,
    horizon_days: int = 1,
    method: Method = "historical",
    seed: int | None = None,
) -> TailRiskResult:
    """Single entry point tools/ calls in M2 (guideline.md §5 worked example).

    Dispatches to `historical_var`/`parametric_var`/`monte_carlo_var`, then
    applies horizon scaling VaR_h = VaR_1 * sqrt(h) under the i.i.d.
    assumption, warning when that assumption is unreliable
    (`horizon_days > HORIZON_SCALING_MAX_RELIABLE_DAYS`).
    """
    if method == "historical":
        base = historical_var(weights, asset_returns, alpha)
    elif method == "parametric":
        base = parametric_var(weights, asset_returns, alpha)
    else:
        base = monte_carlo_var(weights, asset_returns, alpha, seed=seed)

    warnings = list(base.warnings)
    if horizon_days > HORIZON_SCALING_MAX_RELIABLE_DAYS:
        warnings.append(
            f"horizon scaling beyond {HORIZON_SCALING_MAX_RELIABLE_DAYS} days assumes "
            "i.i.d. returns and is unreliable"
        )
    scaled_value = base.value * float(np.sqrt(horizon_days))
    assert_finite(scaled_value, context="portfolio_var")
    return TailRiskResult(
        method=base.method,
        sample_size=base.sample_size,
        value=scaled_value,
        alpha=alpha,
        horizon_days=horizon_days,
        seed=base.seed,
        warnings=warnings,
    )
