from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.beta import beta, downside_beta
from tests.unit.quant.builders import build_market_returns, build_return_matrix, build_weights


def test_single_asset_beta_against_itself_equals_one() -> None:
    returns = build_return_matrix(n_obs=100, tickers=["A"])
    weights = pd.Series({"A": 1.0})
    market_returns = returns["A"]

    result = beta(weights, returns, market_returns)

    assert result.value == pytest.approx(1.0, abs=1e-9)


def test_beta_matches_manual_cov_over_var_calculation() -> None:
    returns = build_return_matrix(n_obs=100)
    weights = build_weights()
    market_returns = build_market_returns(n_obs=100)

    result = beta(weights, returns, market_returns)

    r_p = returns.dot(weights)
    cov_matrix = np.cov(r_p.to_numpy(), market_returns.to_numpy())
    expected = cov_matrix[0, 1] / cov_matrix[1, 1]
    assert result.value == pytest.approx(expected)


def test_downside_beta_only_uses_negative_market_days() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()
    market_returns = build_market_returns(n_obs=300)

    result = downside_beta(weights, returns, market_returns)

    expected_count = int((market_returns < 0).sum())
    assert result.sample_size == expected_count


def test_beta_below_min_observations_raises_insufficient_data_error() -> None:
    returns = build_return_matrix(n_obs=10)
    weights = build_weights()
    market_returns = build_market_returns(n_obs=10)

    with pytest.raises(InsufficientDataError):
        beta(weights, returns, market_returns)


def test_downside_beta_below_min_observations_raises_insufficient_data_error() -> None:
    # Nearly all-positive market returns leave too few down-market days.
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()
    index = build_market_returns(n_obs=300).index
    market_returns = pd.Series([0.01] * 290 + [-0.01] * 10, index=index)

    with pytest.raises(InsufficientDataError):
        downside_beta(weights, returns, market_returns)


def test_beta_raises_on_unaligned_market_index() -> None:
    returns = build_return_matrix(n_obs=100)
    weights = build_weights()
    market_returns = build_market_returns(n_obs=100, seed=999)
    misaligned = market_returns.set_axis(market_returns.index[::-1])

    with pytest.raises(ValueError):
        beta(weights, returns, misaligned)
