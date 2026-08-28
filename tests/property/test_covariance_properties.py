from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quantagent.quant.covariance import (
    correlation_from_covariance,
    ledoit_wolf_covariance,
    portfolio_variance,
)
from tests.property.strategies import bounded_return_matrix, long_only_weights, psd_covariance


@given(data=st.data())
@settings(max_examples=100, deadline=None)
def test_correlation_matrix_is_psd_after_shrinkage(data: st.DataObject) -> None:
    weights = data.draw(long_only_weights(min_n=2, max_n=10))
    returns = data.draw(bounded_return_matrix(n_assets=len(weights), min_t=60, max_t=200))
    returns.columns = weights.index

    cov = ledoit_wolf_covariance(returns)
    correlation = correlation_from_covariance(cov)

    eigenvalues = np.linalg.eigvalsh(correlation.to_numpy())
    assert eigenvalues.min() >= -1e-8


@given(
    data=st.data(),
    epsilon=st.floats(min_value=0.001, max_value=0.05),
    new_asset_variance=st.floats(min_value=0.001, max_value=0.10),
)
@settings(max_examples=100, deadline=None)
def test_adding_uncorrelated_asset_does_not_exceed_weighted_variance_share(
    data: st.DataObject, epsilon: float, new_asset_variance: float
) -> None:
    weights = data.draw(long_only_weights(min_n=2, max_n=8))
    n = len(weights)
    base_matrix = data.draw(psd_covariance(n_assets=n))

    base_variance = portfolio_variance(weights, base_matrix)

    extended_matrix = base_matrix.copy()
    extended_matrix["NEW"] = 0.0
    extended_matrix.loc["NEW"] = 0.0
    extended_matrix.loc["NEW", "NEW"] = new_asset_variance

    renormalized_weights = weights * (1.0 - epsilon)
    extended_weights = renormalized_weights.copy()
    extended_weights["NEW"] = epsilon

    extended_variance = portfolio_variance(extended_weights, extended_matrix)
    upper_bound = (1.0 - epsilon) ** 2 * base_variance + epsilon**2 * new_asset_variance

    assert extended_variance <= upper_bound + 1e-9
