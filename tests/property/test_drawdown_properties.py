from __future__ import annotations

from hypothesis import given, settings

from quantagent.quant.drawdown import max_drawdown
from tests.property.strategies import portfolio


@given(weights_and_returns=portfolio(min_t=250, max_t=400))
@settings(max_examples=100, deadline=None)
def test_drawdown_value_always_in_minus_one_to_zero(weights_and_returns) -> None:
    weights, returns = weights_and_returns

    result = max_drawdown(weights, returns)

    assert -1.0 <= result.value <= 0.0
