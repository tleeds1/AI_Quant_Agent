from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import TRADING_DAYS_PER_YEAR
from quantagent.quant.tracking_error import tracking_error
from tests.unit.quant.builders import build_market_returns, build_return_matrix, build_weights


def test_tracking_error_against_own_benchmark_is_zero() -> None:
    returns = build_return_matrix(n_obs=100, tickers=["A"])
    weights = pd.Series({"A": 1.0})

    result = tracking_error(weights, returns, returns["A"])

    assert result.value == pytest.approx(0.0, abs=1e-12)


def test_tracking_error_matches_hand_computed_annualised_active_volatility() -> None:
    returns = build_return_matrix(n_obs=100)
    weights = build_weights()
    benchmark_returns = build_market_returns(n_obs=100)

    result = tracking_error(weights, returns, benchmark_returns)

    active = returns.dot(weights) - benchmark_returns
    expected = float(np.std(active.to_numpy(), ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    assert result.value == pytest.approx(expected)


def test_tracking_error_is_non_negative_and_reports_sample_size() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()
    benchmark_returns = build_market_returns(n_obs=300)

    result = tracking_error(weights, returns, benchmark_returns)

    assert result.value >= 0.0
    assert result.sample_size == 300
    assert result.method == "annualised_tracking_error"


def test_constant_active_return_gap_has_zero_tracking_error() -> None:
    # A benchmark offset by a fixed daily amount has zero *dispersion* of
    # active return, which is the property that separates TE from alpha.
    returns = build_return_matrix(n_obs=100, tickers=["A"])
    weights = pd.Series({"A": 1.0})
    benchmark_returns = returns["A"] - 0.001

    result = tracking_error(weights, returns, benchmark_returns)

    assert result.value == pytest.approx(0.0, abs=1e-12)


def test_tracking_error_below_min_observations_raises_insufficient_data_error() -> None:
    returns = build_return_matrix(n_obs=10)
    weights = build_weights()
    benchmark_returns = build_market_returns(n_obs=10)

    with pytest.raises(InsufficientDataError):
        tracking_error(weights, returns, benchmark_returns)


def test_tracking_error_raises_on_unaligned_benchmark_index() -> None:
    returns = build_return_matrix(n_obs=100)
    weights = build_weights()
    benchmark_returns = build_market_returns(n_obs=100)
    misaligned = benchmark_returns.set_axis(benchmark_returns.index[::-1])

    with pytest.raises(ValueError):
        tracking_error(weights, returns, misaligned)
