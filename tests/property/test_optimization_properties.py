from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from quantagent.contracts.errors import OptimizationError
from quantagent.quant.optimization import optimize_portfolio
from tests.property.strategies import portfolio


@given(
    pf=portfolio(min_n=3, max_n=6, min_t=300, max_t=500),
    objective=st.sampled_from(["min_variance", "max_utility", "risk_parity"]),
    max_concentration=st.floats(min_value=0.4, max_value=0.8),
    max_turnover=st.floats(min_value=0.2, max_value=0.8),
)
@settings(max_examples=40, deadline=None)
def test_optimization_respects_constraints_properties(
    pf: tuple[pd.Series, pd.DataFrame],
    objective: str,
    max_concentration: float,
    max_turnover: float,
) -> None:
    current_weights, returns = pf

    # Run optimization
    try:
        w_target = optimize_portfolio(
            returns=returns,
            objective=objective,
            current_weights=current_weights,
            max_concentration=max_concentration,
            max_turnover=max_turnover,
            risk_aversion=3.0,
        )
    except OptimizationError:
        # Reject this draw if the solver genuinely found the parameters
        # infeasible -- narrowed to OptimizationError specifically so a
        # real bug elsewhere in optimize_portfolio (a KeyError, an
        # AttributeError) still fails the test instead of being silently
        # discarded as "must have been an infeasible draw."
        assume(False)
        return

    # Assert basic weight validity
    assert np.all(w_target.to_numpy() >= -1e-6)
    assert np.isclose(np.sum(w_target.to_numpy()), 1.0)

    # Assert concentration constraint
    assert np.all(w_target.to_numpy() <= max_concentration + 1e-5)

    # Assert turnover constraint
    turnover = np.sum(
        np.abs(w_target.to_numpy() - current_weights.reindex(w_target.index).fillna(0.0).to_numpy())
    )
    assert turnover <= max_turnover + 1e-4
