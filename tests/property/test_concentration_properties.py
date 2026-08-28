from __future__ import annotations

from hypothesis import given, settings

from quantagent.quant.concentration import portfolio_concentration
from tests.property.strategies import long_only_weights


@given(weights=long_only_weights(min_n=2, max_n=15))
@settings(max_examples=200, deadline=None)
def test_hhi_bounded_between_one_over_n_and_one_for_long_only_weights(weights) -> None:
    result = portfolio_concentration(weights)

    n = len(weights)
    assert 1.0 / n - 1e-9 <= result.hhi <= 1.0 + 1e-9
