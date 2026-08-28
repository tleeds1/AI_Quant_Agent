from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import MIN_CVAR_TAIL_OBSERVATIONS
from quantagent.quant.returns import portfolio_returns
from quantagent.quant.types import TailRiskResult
from quantagent.quant.validation import assert_finite


def historical_cvar(
    weights: pd.Series, asset_returns: pd.DataFrame, alpha: float
) -> TailRiskResult:
    """CVaR_alpha = -E[r_p | r_p <= quantile(r_p, 1-alpha)] (architecture.md §4.4).

    Positive loss fraction. Raises `InsufficientDataError` if fewer than
    `MIN_CVAR_TAIL_OBSERVATIONS` fall in the tail -- refused, never
    approximated by widening alpha.
    """
    r_p = portfolio_returns(weights, asset_returns)
    threshold = np.quantile(r_p.to_numpy(), 1.0 - alpha)
    tail = r_p[r_p <= threshold]
    if len(tail) < MIN_CVAR_TAIL_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_CVAR_TAIL_OBSERVATIONS} tail observations for CVaR, "
            f"got {len(tail)}"
        )
    value = float(-tail.mean())
    assert_finite(value, context="historical_cvar")
    return TailRiskResult(method="historical", sample_size=len(r_p), value=value, alpha=alpha)


def portfolio_cvar(weights: pd.Series, asset_returns: pd.DataFrame, alpha: float) -> TailRiskResult:
    """Portfolio-level CVaR entry point (naming-symmetric with `var.portfolio_var`).

    Historical is the only method defined for M1 -- architecture.md §4.4
    gives no parametric CVaR closed form here.
    """
    return historical_cvar(weights, asset_returns, alpha)
