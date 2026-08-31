# mypy: ignore-errors
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from quantagent.contracts.errors import OptimizationError
from quantagent.quant.covariance import ledoit_wolf_covariance


def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str,
    current_weights: pd.Series | None = None,
    max_concentration: float | None = None,
    max_turnover: float | None = None,
    risk_aversion: float = 2.0,
    target_return: float | None = None,
) -> pd.Series:
    """Optimize portfolio weights using CVXPY based on the given objective and constraints.

    Objectives:
      - "min_variance": Minimize annualized portfolio variance w^T * Sigma * w.
      - "max_utility": Maximize annualized utility w^T * mu - risk_aversion * w^T * Sigma * w.
      - "risk_parity": Solve convex risk parity log-barrier problem and normalize.

    Returns target weights as a pandas Series indexed by asset tickers.
    """
    n_assets = len(returns.columns)
    if n_assets == 0:
        raise OptimizationError("No assets to optimize.")

    # Calculate covariance and mean returns
    cov_res = ledoit_wolf_covariance(returns)
    sigma = cov_res.matrix.to_numpy() * 252  # Annualized Ledoit-Wolf covariance
    mu = returns.mean().to_numpy() * 252  # Annualized mean return

    # Check current weights
    if current_weights is not None:
        w0 = current_weights.reindex(returns.columns).fillna(0.0).to_numpy()
    else:
        w0 = np.zeros(n_assets)

    if objective == "risk_parity":
        # Convex log-barrier formulation of risk parity:
        # Minimize 0.5 * x^T * Sigma * x - sum(log(x))
        # After solving, target weights are x / sum(x)
        x = cp.Variable(n_assets)
        constraints = [x >= 1e-9]

        if max_concentration is not None:
            # w_i = x_i / cp.sum(x) <= max_concentration
            constraints.append(x <= max_concentration * cp.sum(x))

        if max_turnover is not None and current_weights is not None:
            # sum(|w_i - w0_i|) <= max_turnover
            # cp.sum(cp.abs(x - w0 * cp.sum(x))) <= max_turnover * cp.sum(x)
            constraints.append(cp.sum(cp.abs(x - w0 * cp.sum(x))) <= max_turnover * cp.sum(x))

        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(x, sigma) - cp.sum(cp.log(x))), constraints
        )
        try:
            prob.solve()
        except Exception as e:
            raise OptimizationError(f"Optimization failed to solve: {e}") from e

        if x.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise OptimizationError(f"Optimization failed: status={prob.status}")

        weights = x.value / np.sum(x.value)
    else:
        # Standard min_variance or max_utility (mean-variance)
        w = cp.Variable(n_assets)
        constraints = [w >= 0, cp.sum(w) == 1.0]

        if max_concentration is not None:
            constraints.append(w <= max_concentration)

        if max_turnover is not None and current_weights is not None:
            constraints.append(cp.sum(cp.abs(w - w0)) <= max_turnover)

        if target_return is not None:
            constraints.append(w @ mu >= target_return)

        if objective == "min_variance":
            obj = cp.Minimize(cp.quad_form(w, sigma))
        elif objective == "max_utility":
            obj = cp.Maximize(w @ mu - risk_aversion * cp.quad_form(w, sigma))
        else:
            raise OptimizationError(f"Unknown objective: {objective}")

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve()
        except Exception as e:
            raise OptimizationError(f"Optimization failed to solve: {e}") from e

        if w.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise OptimizationError(f"Optimization failed: status={prob.status}")

        weights = w.value

    # Standardize weights: clip slightly negative values due to solver precision and re-normalize
    weights = np.clip(weights, 0.0, None)
    weights /= np.sum(weights)

    return pd.Series(weights, index=returns.columns)
