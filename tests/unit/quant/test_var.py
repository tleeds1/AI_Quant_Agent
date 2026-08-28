from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import MIN_VAR_OBSERVATIONS
from quantagent.quant.var import historical_var, monte_carlo_var, parametric_var, portfolio_var
from tests.unit.quant.builders import build_return_matrix, build_weights


def test_historical_var_matches_manual_quantile_calculation() -> None:
    # 10 hand-picked points followed by flat padding (to clear MIN_VAR_OBSERVATIONS)
    # appended at the tail of the return series -- the expected value is computed
    # from the exact same full series historical_var() sees.
    hand_picked = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.01, -0.03, 0.02, -0.01]
    padding = [0.0005] * (MIN_VAR_OBSERVATIONS - len(hand_picked) + 5)
    values_a = hand_picked + padding
    values_b = [0.5 * v for v in values_a]
    index = pd.bdate_range("2020-01-01", periods=len(values_a))
    returns = pd.DataFrame({"A": values_a, "B": values_b}, index=index)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    r_p = 0.5 * returns["A"] + 0.5 * returns["B"]

    result = historical_var(weights, returns, alpha=0.90)

    expected = -np.quantile(r_p.to_numpy(), 0.10)
    assert result.value == pytest.approx(expected)
    assert result.method == "historical"


def test_historical_var_below_min_observations_raises_insufficient_data_error() -> None:
    returns = build_return_matrix(n_obs=MIN_VAR_OBSERVATIONS - 1)
    weights = build_weights()

    with pytest.raises(InsufficientDataError):
        historical_var(weights, returns, alpha=0.95)


def test_parametric_var_matches_scipy_norm_ppf_closed_form() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()

    result = parametric_var(weights, returns, alpha=0.95)

    from quantagent.quant.covariance import ledoit_wolf_covariance, portfolio_variance
    from quantagent.quant.returns import portfolio_returns

    r_p = portfolio_returns(weights, returns)
    cov = ledoit_wolf_covariance(returns)
    mu_p = r_p.mean()
    sigma_p = np.sqrt(portfolio_variance(weights, cov.matrix))
    z = stats.norm.ppf(0.05)
    expected = -(mu_p + z * sigma_p)

    assert result.value == pytest.approx(expected, abs=1e-9)


def test_zero_variance_portfolio_var_is_zero() -> None:
    index = pd.bdate_range("2020-01-01", periods=300)
    constant_returns = pd.DataFrame({"A": [0.0005] * 300, "B": [0.0005] * 300}, index=index)
    weights = pd.Series({"A": 0.5, "B": 0.5})

    hist = historical_var(weights, constant_returns, alpha=0.95)
    param = parametric_var(weights, constant_returns, alpha=0.95)

    assert hist.value == pytest.approx(-0.0005, abs=1e-9)
    assert param.value == pytest.approx(-0.0005, abs=1e-9)


def test_monte_carlo_var_is_reproducible_with_same_seed() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()

    first = monte_carlo_var(weights, returns, alpha=0.95, seed=42)
    second = monte_carlo_var(weights, returns, alpha=0.95, seed=42)

    assert first.value == second.value
    assert first.seed == 42
    assert second.seed == 42


def test_monte_carlo_var_echoes_generated_seed_when_none_passed() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()

    result = monte_carlo_var(weights, returns, alpha=0.95, seed=None)

    assert result.seed is not None
    replay = monte_carlo_var(weights, returns, alpha=0.95, seed=result.seed)
    assert replay.value == result.value


def test_monte_carlo_var_below_min_simulations_raises_value_error() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()

    with pytest.raises(ValueError):
        monte_carlo_var(weights, returns, alpha=0.95, n_sims=100, seed=1)


def test_monte_carlo_var_bootstrap_innovation_runs() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()

    result = monte_carlo_var(weights, returns, alpha=0.95, innovation="bootstrap", seed=1)

    assert result.method == "monte_carlo_bootstrap"
    assert np.isfinite(result.value)


def test_portfolio_var_horizon_scaling_matches_sqrt_h_manually() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()

    var_1d = portfolio_var(weights, returns, alpha=0.95, horizon_days=1)
    var_5d = portfolio_var(weights, returns, alpha=0.95, horizon_days=5)

    assert var_5d.value == pytest.approx(var_1d.value * (5**0.5), abs=1e-9)


def test_portfolio_var_warns_when_horizon_exceeds_10_days() -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()

    result = portfolio_var(weights, returns, alpha=0.95, horizon_days=15)

    assert any("horizon scaling" in w for w in result.warnings)


@pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo"])
def test_portfolio_var_dispatches_to_correct_method(method: str) -> None:
    returns = build_return_matrix(n_obs=300)
    weights = build_weights()

    result = portfolio_var(weights, returns, alpha=0.95, method=method, seed=1)  # type: ignore[arg-type]

    assert result.method.startswith(method if method != "monte_carlo" else "monte_carlo")
