from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.quant.returns import (
    annualize_return,
    annualize_volatility,
    log_returns,
    portfolio_returns,
    simple_returns,
    simulate_equity_curve,
)


def _prices(values: list[float]) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.DataFrame({"AAA": values}, index=index)


def test_simple_returns_matches_hand_computed_values() -> None:
    prices = _prices([100.0, 110.0, 99.0, 108.9])

    result = simple_returns(prices)["AAA"].to_numpy()

    assert result == pytest.approx([0.10, -0.10, 0.10], abs=1e-9)


def test_log_returns_matches_hand_computed_values() -> None:
    prices = _prices([100.0, 110.0, 121.0])

    result = log_returns(prices)["AAA"].to_numpy()

    expected = [np.log(110.0 / 100.0), np.log(121.0 / 110.0)]
    assert result == pytest.approx(expected, abs=1e-9)


def test_portfolio_returns_matches_manual_weighted_sum() -> None:
    index = pd.bdate_range("2020-01-01", periods=3)
    asset_returns = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [-0.02, 0.00, 0.03]}, index=index)
    weights = pd.Series({"A": 0.6, "B": 0.4})

    result = portfolio_returns(weights, asset_returns)

    expected = 0.6 * asset_returns["A"] + 0.4 * asset_returns["B"]
    assert result.to_numpy() == pytest.approx(expected.to_numpy())


def test_portfolio_returns_uses_current_weights_not_historical_nav() -> None:
    index = pd.bdate_range("2020-01-01", periods=2)
    asset_returns = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=index)
    current_weights = pd.Series({"A": 1.0, "B": 0.0})

    result = portfolio_returns(current_weights, asset_returns)

    # Current weights (100% A) applied throughout -> [1.0, 0.0], not a NAV-based
    # answer that would reflect A's outperformance shifting weight toward B.
    assert result.to_numpy() == pytest.approx([1.0, 0.0])


def test_portfolio_returns_raises_on_index_mismatch() -> None:
    index = pd.bdate_range("2020-01-01", periods=2)
    asset_returns = pd.DataFrame({"A": [0.01, 0.02]}, index=index)
    weights = pd.Series({"A": 0.5, "B": 0.5})

    with pytest.raises(ValueError):
        portfolio_returns(weights, asset_returns)


def test_simulate_equity_curve_matches_manual_cumprod() -> None:
    index = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.Series([0.10, -0.10, 0.05], index=index)

    curve = simulate_equity_curve(returns, initial_value=100.0)

    expected = [110.0, 99.0, 103.95]
    assert curve.to_numpy() == pytest.approx(expected)


def test_annualize_return_matches_manual_formula() -> None:
    index = pd.bdate_range("2020-01-01", periods=3)
    returns = pd.Series([0.001, 0.001, 0.001], index=index)

    result = annualize_return(returns, periods_per_year=252)

    assert result == pytest.approx((1.001) ** 252 - 1)


def test_annualize_volatility_matches_manual_formula() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005], index=index)

    result = annualize_volatility(returns, periods_per_year=252)

    assert result == pytest.approx(returns.std(ddof=1) * (252**0.5))
