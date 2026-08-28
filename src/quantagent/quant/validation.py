from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def as_float64(values: pd.Series | pd.DataFrame | NDArray[np.floating]) -> NDArray[np.float64]:
    """Coerce array-like input to a float64 numpy array (guideline.md §6 rule 1:
    float64 everywhere, never float32, for money or risk).
    """
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return values.to_numpy(dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def assert_finite(value: float, *, context: str) -> None:
    """Raise `ValueError` if `value` is NaN or infinite.

    The single call site every quant/ function uses immediately before
    returning a scalar metric (guideline.md §6 rule 5).
    """
    if not np.isfinite(value):
        raise ValueError(f"{context}: computed value is not finite ({value!r})")


def assert_no_nan(data: pd.Series | pd.DataFrame, *, context: str) -> None:
    """Raise `ValueError` if `data` contains any NaN.

    quant/ never silently drops rows this deep in the pipeline -- NaN
    handling belongs to `calendar.align_calendars`, run upstream by the
    caller before any cross-asset computation.
    """
    if bool(pd.isna(data).to_numpy().any()):
        raise ValueError(f"{context}: input contains NaN; run calendar.align_calendars first")


def assert_weights_sum_to_one(weights: pd.Series, *, tolerance: float = 1e-6) -> None:
    """Raise `ValueError` if `weights` doesn't sum to 1 within `tolerance`."""
    total = float(weights.sum())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"weights must sum to 1.0 (within {tolerance}), got {total}")


def assert_aligned_index(
    a: pd.Series | pd.DataFrame, b: pd.Series | pd.DataFrame, *, context: str
) -> None:
    """Raise `ValueError` if `a` and `b` don't share an identical index.

    Catches a caller passing an unaligned panel before it silently
    misaligns dates (guideline.md §6 rule 6).
    """
    if not a.index.equals(b.index):
        raise ValueError(f"{context}: inputs are not index-aligned; align calendars first")


def assert_single_currency(currencies: Sequence[str]) -> None:
    """Raise `ValueError` if `currencies` contains more than one distinct value.

    quant/ receives no currency metadata in M1 (v1 single-base-currency
    assumption, guideline.md §6 rule 7) -- this guard is exported for the
    first future caller (data/ or tools/) that threads currency codes
    through, so the enforcement point exists and is tested ahead of that
    need rather than bolted on later.
    """
    distinct = set(currencies)
    if len(distinct) > 1:
        raise ValueError(f"mixed currencies are not supported in v1, got {sorted(distinct)}")
