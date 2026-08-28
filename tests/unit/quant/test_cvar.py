from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.cvar import historical_cvar, portfolio_cvar
from quantagent.quant.var import historical_var
from tests.unit.quant.builders import build_return_matrix, build_weights


def test_historical_cvar_matches_manual_tail_mean() -> None:
    n_obs = 150  # alpha=0.80 -> ~20% tail -> ~30 observations, clears MIN_CVAR_TAIL_OBSERVATIONS
    index = pd.bdate_range("2020-01-01", periods=n_obs)
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.02, size=n_obs)
    returns = pd.DataFrame({"A": values, "B": values}, index=index)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    r_p = returns.dot(weights)

    result = historical_cvar(weights, returns, alpha=0.80)

    threshold = np.quantile(r_p.to_numpy(), 0.20)
    expected = -r_p[r_p <= threshold].mean()
    assert result.value == pytest.approx(expected)


def test_cvar_below_min_tail_observations_raises_insufficient_data_error() -> None:
    returns = build_return_matrix(n_obs=50)
    weights = build_weights()

    with pytest.raises(InsufficientDataError):
        historical_cvar(weights, returns, alpha=0.999)


def test_cvar_is_never_less_than_var_on_fixed_dataset() -> None:
    returns = build_return_matrix(n_obs=500)
    weights = build_weights()

    var_result = historical_var(weights, returns, alpha=0.95)
    cvar_result = historical_cvar(weights, returns, alpha=0.95)

    assert cvar_result.value >= var_result.value


def test_portfolio_cvar_matches_historical_cvar() -> None:
    returns = build_return_matrix(n_obs=500)
    weights = build_weights()

    assert (
        portfolio_cvar(weights, returns, alpha=0.95).value
        == historical_cvar(weights, returns, alpha=0.95).value
    )
