from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantagent.quant.beta import beta
from quantagent.quant.concentration import portfolio_concentration
from quantagent.quant.cvar import historical_cvar
from quantagent.quant.drawdown import max_drawdown
from quantagent.quant.var import historical_var
from tests.property.strategies import portfolio


@given(
    weights_and_returns=portfolio(min_n=2, max_n=8, min_t=300, max_t=400),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=None)
def test_reordering_holdings_does_not_change_portfolio_level_metrics(
    weights_and_returns, seed: int
) -> None:
    weights, returns = weights_and_returns
    rng = np.random.default_rng(seed)
    permuted_tickers = list(weights.index)
    rng.shuffle(permuted_tickers)
    permuted_weights = weights.reindex(permuted_tickers)
    permuted_returns = returns[permuted_tickers]
    market_returns = pd.Series(
        np.random.default_rng(seed + 1).uniform(-0.05, 0.05, size=len(returns)),
        index=returns.index,
    )

    original_var = historical_var(weights, returns, alpha=0.90)
    permuted_var = historical_var(permuted_weights, permuted_returns, alpha=0.90)
    assert original_var.value == pytest.approx(permuted_var.value)

    original_cvar = historical_cvar(weights, returns, alpha=0.90)
    permuted_cvar = historical_cvar(permuted_weights, permuted_returns, alpha=0.90)
    assert original_cvar.value == pytest.approx(permuted_cvar.value)

    original_beta = beta(weights, returns, market_returns)
    permuted_beta = beta(permuted_weights, permuted_returns, market_returns)
    assert original_beta.value == pytest.approx(permuted_beta.value)

    original_hhi = portfolio_concentration(weights).hhi
    permuted_hhi = portfolio_concentration(permuted_weights).hhi
    assert original_hhi == pytest.approx(permuted_hhi)

    original_drawdown = max_drawdown(weights, returns)
    permuted_drawdown = max_drawdown(permuted_weights, permuted_returns)
    assert original_drawdown.value == pytest.approx(permuted_drawdown.value)
