from __future__ import annotations

import pandas as pd
import pytest

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.drawdown import max_drawdown
from tests.unit.quant.builders import build_return_matrix, build_weights


def _single_asset_returns_from_curve(curve: list[float]) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=len(curve))
    prices = pd.DataFrame({"A": curve}, index=index)
    returns = prices.pct_change().iloc[1:]
    return returns


def test_max_drawdown_matches_hand_computed_value_on_known_equity_curve() -> None:
    curve = [100.0, 120.0, 90.0, 95.0, 130.0]
    returns = _single_asset_returns_from_curve(curve)
    # Pad with additional flat (zero-return) history so the sample-size guard passes,
    # without changing the drawdown computed on the meaningful window.
    padding_index = pd.bdate_range("2019-01-01", periods=250)
    padding = pd.DataFrame({"A": [0.0] * 250}, index=padding_index)
    full_returns = pd.concat([padding, returns])
    weights = pd.Series({"A": 1.0})

    result = max_drawdown(weights, full_returns, initial_value=100.0)

    assert result.value == pytest.approx(90.0 / 120.0 - 1.0)


def test_recovery_date_is_none_when_curve_never_recovers() -> None:
    curve = [100.0, 120.0, 90.0, 95.0]
    returns = _single_asset_returns_from_curve(curve)
    padding_index = pd.bdate_range("2019-01-01", periods=250)
    padding = pd.DataFrame({"A": [0.0] * 250}, index=padding_index)
    full_returns = pd.concat([padding, returns])
    weights = pd.Series({"A": 1.0})

    result = max_drawdown(weights, full_returns, initial_value=100.0)

    assert result.recovery_date is None
    assert result.recovery_duration_days is None


def test_recovery_date_and_duration_correct_when_curve_recovers() -> None:
    curve = [100.0, 120.0, 90.0, 95.0, 130.0]
    returns = _single_asset_returns_from_curve(curve)
    padding_index = pd.bdate_range("2019-01-01", periods=250)
    padding = pd.DataFrame({"A": [0.0] * 250}, index=padding_index)
    full_returns = pd.concat([padding, returns])
    weights = pd.Series({"A": 1.0})

    result = max_drawdown(weights, full_returns, initial_value=100.0)

    assert result.recovery_date is not None
    assert result.recovery_duration_days is not None
    assert result.recovery_duration_days > 0


def test_drawdown_below_min_observations_raises_insufficient_data_error() -> None:
    returns = build_return_matrix(n_obs=10)
    weights = build_weights()

    with pytest.raises(InsufficientDataError):
        max_drawdown(weights, returns)


def test_drawdown_value_is_never_positive() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()

    result = max_drawdown(weights, returns)

    assert result.value <= 0.0
