from __future__ import annotations

import pandas as pd
import pytest

from quantagent.quant.concentration import portfolio_concentration


def test_hhi_matches_manual_sum_of_squares() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})

    result = portfolio_concentration(weights)

    expected_hhi = 0.4**2 + 0.3**2 + 0.2**2 + 0.1**2
    assert result.hhi == pytest.approx(expected_hhi)


def test_effective_holdings_equals_inverse_hhi() -> None:
    weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})

    result = portfolio_concentration(weights)

    assert result.effective_holdings == pytest.approx(1.0 / result.hhi)


def test_uniform_weights_give_hhi_equal_to_one_over_n() -> None:
    weights = pd.Series({f"T{i}": 0.2 for i in range(5)})

    result = portfolio_concentration(weights)

    assert result.hhi == pytest.approx(0.2)


def test_single_holding_gives_hhi_equal_to_one() -> None:
    weights = pd.Series({"A": 1.0})

    result = portfolio_concentration(weights)

    assert result.hhi == pytest.approx(1.0)
    assert result.effective_holdings == pytest.approx(1.0)


def test_top_n_weight_sums_largest_positions() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.2, "C": 0.15, "D": 0.1, "E": 0.05})

    result = portfolio_concentration(weights, top_n=2)

    assert result.top_n_weight == pytest.approx(0.7)


def test_weights_not_summing_to_one_raises_value_error() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.3})

    with pytest.raises(ValueError):
        portfolio_concentration(weights)
