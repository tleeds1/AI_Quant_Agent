from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from quantagent.quant.component_var import parametric_component_var
from quantagent.quant.types import CovarianceResult
from tests.property.strategies import bounded_return_matrix, long_only_weights, psd_covariance


@given(data=st.data())
@settings(max_examples=200, deadline=None)
def test_parametric_component_var_sums_to_portfolio_var(data: st.DataObject) -> None:
    weights = data.draw(long_only_weights(min_n=2, max_n=8))
    n = len(weights)
    matrix = data.draw(psd_covariance(n_assets=n))
    returns = data.draw(bounded_return_matrix(n_assets=n, min_t=300, max_t=400))
    returns.columns = weights.index
    cov = CovarianceResult(
        method="hypothesis_psd",
        sample_size=len(returns),
        matrix=matrix,
        shrinkage_intensity=0.0,
        t_over_n_ratio=len(returns) / n,
        n_assets=n,
    )

    result = parametric_component_var(weights, returns, alpha=0.95, cov=cov)

    total = sum(result.components.values())
    tolerance = max(1e-9, abs(result.portfolio_value) * 1e-6)
    assert abs(total - result.portfolio_value) < tolerance
