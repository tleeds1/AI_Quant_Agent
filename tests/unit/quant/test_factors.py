from __future__ import annotations

import pytest
import statsmodels.api as sm

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.factors import factor_exposure
from tests.unit.quant.builders import build_factor_returns, build_return_matrix


def _portfolio_excess_returns(n_obs: int = 200):
    returns = build_return_matrix(n_obs=n_obs, tickers=["A"])
    return returns["A"]


def test_factor_exposure_matches_statsmodels_ols_directly() -> None:
    portfolio_returns = _portfolio_excess_returns(200)
    factors = build_factor_returns(n_obs=200)

    result = factor_exposure(portfolio_returns, factors)

    design = sm.add_constant(factors)
    reference = sm.OLS(portfolio_returns, design).fit(
        cov_type="HAC", cov_kwds={"maxlags": result.hac_lags}
    )

    for name in factors.columns:
        assert result.betas[name] == pytest.approx(reference.params[name])
        assert result.t_stats[name] == pytest.approx(reference.tvalues[name])
    assert result.r_squared == pytest.approx(reference.rsquared)


def test_hac_standard_errors_differ_from_ols_standard_errors_on_autocorrelated_series() -> None:
    portfolio_returns = _portfolio_excess_returns(200)
    # Introduce autocorrelation via a rolling mean so HAC vs. plain-OLS SEs diverge.
    autocorrelated = portfolio_returns.rolling(3, min_periods=1).mean()
    factors = build_factor_returns(n_obs=200)

    design = sm.add_constant(factors)
    ols_fit = sm.OLS(autocorrelated, design).fit()
    hac_fit = sm.OLS(autocorrelated, design).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    assert ols_fit.bse.to_numpy() != pytest.approx(hac_fit.bse.to_numpy())


def test_low_t_stat_factor_flagged_as_not_significant() -> None:
    portfolio_returns = _portfolio_excess_returns(200)
    factors = build_factor_returns(n_obs=200)

    result = factor_exposure(portfolio_returns, factors)

    for name in factors.columns:
        expected_significant = abs(result.t_stats[name]) >= 2.0
        assert result.significant[name] == expected_significant


def test_factor_exposure_below_min_observations_raises_insufficient_data_error() -> None:
    portfolio_returns = _portfolio_excess_returns(50)
    factors = build_factor_returns(n_obs=50)

    with pytest.raises(InsufficientDataError):
        factor_exposure(portfolio_returns, factors)


def test_underdetermined_regression_raises_insufficient_data_error() -> None:
    portfolio_returns = _portfolio_excess_returns(65)
    factors = build_factor_returns(n_obs=65, factor_names=[f"f{i}" for i in range(10)])

    with pytest.raises(InsufficientDataError):
        factor_exposure(portfolio_returns, factors)


def test_hac_lags_defaults_to_newey_west_rule_when_omitted() -> None:
    import math

    portfolio_returns = _portfolio_excess_returns(200)
    factors = build_factor_returns(n_obs=200)

    result = factor_exposure(portfolio_returns, factors)

    assert result.hac_lags == math.floor(4 * (200 / 100) ** (2 / 9))
