from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from quantagent.quant.cvar import historical_cvar
from quantagent.quant.var import historical_var
from tests.property.strategies import portfolio

# alpha range + min_t=300 keeps the CVaR tail comfortably above
# MIN_CVAR_TAIL_OBSERVATIONS=20 (worst case ~300*0.20=60 tail observations).
_ALPHA = st.floats(min_value=0.80, max_value=0.90)


@given(weights_and_returns=portfolio(), alpha=_ALPHA)
@settings(max_examples=100, deadline=None)
def test_cvar_always_at_least_as_large_as_var(weights_and_returns, alpha: float) -> None:
    weights, returns = weights_and_returns

    var_result = historical_var(weights, returns, alpha)
    cvar_result = historical_cvar(weights, returns, alpha)

    assert cvar_result.value >= var_result.value - 1e-12


@given(weights_and_returns=portfolio(), alpha_low=_ALPHA, alpha_delta=st.floats(0.001, 0.05))
@settings(max_examples=100, deadline=None)
def test_var_is_monotone_non_increasing_in_alpha(
    weights_and_returns, alpha_low: float, alpha_delta: float
) -> None:
    weights, returns = weights_and_returns
    alpha_high = min(alpha_low + alpha_delta, 0.95)

    var_low = historical_var(weights, returns, alpha_low)
    var_high = historical_var(weights, returns, alpha_high)

    assert var_high.value >= var_low.value - 1e-12
