from __future__ import annotations

import pandas as pd

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import MIN_VAR_OBSERVATIONS
from quantagent.quant.returns import portfolio_returns, simulate_equity_curve
from quantagent.quant.types import DrawdownResult
from quantagent.quant.validation import assert_finite


def max_drawdown(
    weights: pd.Series, asset_returns: pd.DataFrame, *, initial_value: float = 1.0
) -> DrawdownResult:
    """MDD = min_t (V_t / max_(s<=t) V_s - 1) on the current-weight simulated
    equity curve (architecture.md §4.4). `value` <= 0 always. Reports the
    peak date (most recent running-max date before the trough), the trough
    date, the recovery date (first date the curve re-exceeds the pre-
    drawdown peak) or `None` if not yet recovered, and the recovery
    duration in days (`None` if unrecovered). Raises `InsufficientDataError`
    below `MIN_VAR_OBSERVATIONS` -- a drawdown estimate needs the same
    minimum history as VaR.
    """
    r_p = portfolio_returns(weights, asset_returns)
    if len(r_p) < MIN_VAR_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_VAR_OBSERVATIONS} observations for max drawdown, "
            f"got {len(r_p)}"
        )
    curve = simulate_equity_curve(r_p, initial_value=initial_value)
    running_max = curve.cummax()
    drawdown_series = curve / running_max - 1.0

    trough_idx = pd.Timestamp(drawdown_series.idxmin())
    mdd = float(drawdown_series.loc[trough_idx])
    assert_finite(mdd, context="max_drawdown")

    peak_value = float(running_max.loc[trough_idx])
    peak_idx = pd.Timestamp(curve.loc[:trough_idx][curve.loc[:trough_idx] == peak_value].index[-1])
    recovery_idx, recovery_days = _find_recovery(curve, trough_idx, peak_value)

    return DrawdownResult(
        method="current_weight_equity_curve",
        sample_size=len(r_p),
        value=mdd,
        peak_date=pd.Timestamp(peak_idx).date(),
        trough_date=pd.Timestamp(trough_idx).date(),
        recovery_date=pd.Timestamp(recovery_idx).date() if recovery_idx is not None else None,
        recovery_duration_days=recovery_days,
    )


def _find_recovery(
    curve: pd.Series, trough_idx: pd.Timestamp, peak_value: float
) -> tuple[pd.Timestamp | None, int | None]:
    """First date after `trough_idx` where `curve` re-exceeds `peak_value`."""
    after_trough = curve.loc[curve.index > trough_idx]
    recovered = after_trough[after_trough >= peak_value]
    if recovered.empty:
        return None, None
    recovery_idx = recovered.index[0]
    days = (pd.Timestamp(recovery_idx) - pd.Timestamp(trough_idx)).days
    return recovery_idx, int(days)
