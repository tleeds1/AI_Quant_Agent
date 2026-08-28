from __future__ import annotations

import pandas as pd

from quantagent.quant.constants import CALENDAR_MAX_FORWARD_FILL_DAYS


def align_calendars(
    prices: pd.DataFrame, *, max_forward_fill_days: int = CALENDAR_MAX_FORWARD_FILL_DAYS
) -> tuple[pd.DataFrame, list[str]]:
    """Align a multi-ticker price panel onto a single, gap-free trading calendar.

    `prices` is expected to already be indexed on the union of trading days
    observed across `prices.columns` (each column may have NaN on days that
    ticker didn't trade). Fill rule: a missing value is forward-filled for up
    to `max_forward_fill_days` consecutive rows per column (covers a single
    exchange holiday or a late listing); anything still missing after that is
    dropped as a row. Must run before any cross-asset computation --
    correlation, covariance, portfolio returns -- a misaligned panel silently
    corrupts every one of them (guideline.md §6 rule 6).

    Returns the aligned panel and a list of human-readable warnings describing
    what was filled or dropped (NaN policy: drop-with-warning, guideline.md
    §6 rule 5).
    """
    sorted_prices = prices.sort_index()
    filled = sorted_prices.ffill(limit=max_forward_fill_days)

    warnings: list[str] = []
    filled_counts = sorted_prices.isna().sum() - filled.isna().sum()
    for ticker, count in filled_counts.items():
        if count > 0:
            warnings.append(f"{ticker}: forward-filled {int(count)} gap(s)")

    still_missing = filled.isna().any(axis=1)
    if still_missing.any():
        dropped_tickers = filled.columns[filled.isna().any(axis=0)].tolist()
        warnings.append(
            f"dropped {int(still_missing.sum())} row(s) with unfillable gaps "
            f"(tickers involved: {dropped_tickers})"
        )

    aligned = filled.loc[~still_missing]
    return aligned, warnings
