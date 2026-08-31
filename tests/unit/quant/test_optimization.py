from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.contracts.errors import OptimizationError
from quantagent.quant.optimization import optimize_portfolio


@pytest.fixture
def dummy_returns() -> pd.DataFrame:
    np.random.seed(42)
    # 504 rows, 3 assets
    data = np.random.normal(0.0002, 0.01, size=(504, 3))
    # Add trend to first asset so max_utility prefers it
    data[:, 0] += 0.001
    return pd.DataFrame(data, columns=["AAPL", "MSFT", "GOOG"])


def test_min_variance_weights(dummy_returns: pd.DataFrame) -> None:
    weights = optimize_portfolio(dummy_returns, objective="min_variance")

    assert isinstance(weights, pd.Series)
    assert list(weights.index) == ["AAPL", "MSFT", "GOOG"]
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), 1.0)


def test_max_utility_weights(dummy_returns: pd.DataFrame) -> None:
    # Asset 0 (AAPL) has higher return trend, should be weighted heavily in max_utility
    weights = optimize_portfolio(dummy_returns, objective="max_utility", risk_aversion=1.0)

    assert weights["AAPL"] > weights["MSFT"]
    assert np.isclose(np.sum(weights), 1.0)


def test_risk_parity_weights(dummy_returns: pd.DataFrame) -> None:
    weights = optimize_portfolio(dummy_returns, objective="risk_parity")

    assert np.all(weights > 0.0)
    assert np.isclose(np.sum(weights), 1.0)


def test_concentration_constraint(dummy_returns: pd.DataFrame) -> None:
    # Cap concentration at 40%
    weights = optimize_portfolio(dummy_returns, objective="max_utility", max_concentration=0.4)

    assert np.all(weights <= 0.40001)
    assert np.isclose(np.sum(weights), 1.0)


def test_turnover_constraint(dummy_returns: pd.DataFrame) -> None:
    # Initial weights: equal weight (0.33, 0.33, 0.33)
    current_weights = pd.Series([0.333, 0.333, 0.334], index=["AAPL", "MSFT", "GOOG"])

    # Restrict turnover to 10%
    weights = optimize_portfolio(
        dummy_returns,
        objective="max_utility",
        current_weights=current_weights,
        max_turnover=0.1,
    )

    turnover = np.sum(np.abs(weights - current_weights))
    assert turnover <= 0.10001
    assert np.isclose(np.sum(weights), 1.0)


def test_optimization_failure_infeasible(dummy_returns: pd.DataFrame) -> None:
    # Set an impossible target return constraint, e.g. 500% annualized
    with pytest.raises(OptimizationError):
        optimize_portfolio(
            dummy_returns,
            objective="min_variance",
            target_return=5.0,
        )


def test_unknown_objective(dummy_returns: pd.DataFrame) -> None:
    with pytest.raises(OptimizationError):
        optimize_portfolio(dummy_returns, objective="invalid_objective")
