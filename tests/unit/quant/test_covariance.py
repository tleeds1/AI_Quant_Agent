from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.covariance import (
    correlation_from_covariance,
    ledoit_wolf_covariance,
    portfolio_variance,
)
from tests.unit.quant.builders import build_return_matrix


def test_ledoit_wolf_covariance_matches_sklearn_directly() -> None:
    returns = build_return_matrix(n_obs=100)

    result = ledoit_wolf_covariance(returns)

    reference = LedoitWolf().fit(returns.to_numpy())
    assert np.allclose(result.matrix.to_numpy(), reference.covariance_)
    assert result.shrinkage_intensity == pytest.approx(reference.shrinkage_)


def test_covariance_below_min_observations_raises_insufficient_data_error() -> None:
    returns = build_return_matrix(n_obs=10)

    with pytest.raises(InsufficientDataError):
        ledoit_wolf_covariance(returns)


def test_t_over_n_ratio_computed_correctly() -> None:
    returns = build_return_matrix(n_obs=100, tickers=["A", "B", "C", "D"])

    result = ledoit_wolf_covariance(returns)

    assert result.t_over_n_ratio == pytest.approx(100 / 4)
    assert result.n_assets == 4


def test_portfolio_variance_matches_manual_quadratic_form() -> None:
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    weights = pd.Series({"A": 0.6, "B": 0.4})

    result = portfolio_variance(weights, cov)

    w = np.array([0.6, 0.4])
    expected = w @ cov.to_numpy() @ w
    assert result == pytest.approx(expected)


def test_correlation_from_covariance_diagonal_is_one() -> None:
    returns = build_return_matrix(n_obs=100)
    cov = ledoit_wolf_covariance(returns)

    correlation = correlation_from_covariance(cov)

    assert np.allclose(np.diag(correlation.to_numpy()), 1.0)
