from __future__ import annotations

import pandas as pd
import pytest

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.component_var import historical_component_var, parametric_component_var
from quantagent.quant.types import CovarianceResult
from tests.unit.quant.builders import build_return_matrix, build_weights


def test_parametric_component_var_sums_to_portfolio_var_on_hand_built_covariance() -> None:
    tickers = ["A", "B", "C"]
    matrix = pd.DataFrame(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ],
        index=tickers,
        columns=tickers,
    )
    cov = CovarianceResult(
        method="hand_built",
        sample_size=300,
        matrix=matrix,
        shrinkage_intensity=0.0,
        t_over_n_ratio=100.0,
        n_assets=3,
    )
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    asset_returns = build_return_matrix(n_obs=300, tickers=tickers)

    result = parametric_component_var(weights, asset_returns, alpha=0.95, cov=cov)

    assert sum(result.components.values()) == pytest.approx(result.portfolio_value, abs=1e-9)


def test_historical_component_var_matches_manual_conditional_mean() -> None:
    tickers = ["A", "B"]
    asset_returns = build_return_matrix(n_obs=300, tickers=tickers)
    weights = pd.Series({"A": 0.5, "B": 0.5})

    result = historical_component_var(weights, asset_returns, alpha=0.90)

    r_p = asset_returns.dot(weights)
    threshold = r_p.quantile(0.10)
    tail_mask = r_p <= threshold
    expected_a = -(weights["A"] * asset_returns.loc[tail_mask, "A"].mean())
    assert result.components["A"] == pytest.approx(expected_a)


def test_historical_component_var_below_min_tail_raises_insufficient_data() -> None:
    returns = build_return_matrix(n_obs=50)
    weights = build_weights()

    with pytest.raises(InsufficientDataError):
        historical_component_var(weights, returns, alpha=0.999)
