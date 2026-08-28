from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quantagent.quant.types import ScalarResult
from quantagent.quant.validation import assert_finite, assert_no_nan


def apply_stress_scenario(weights: pd.Series, shocks: Mapping[str, float]) -> ScalarResult:
    """P&L = sum_i w_i * shocks.get(i, 0.0) for an already-parsed shock vector
    (e.g. `{"AAPL": -0.12, "NVDA": -0.35}`). Pure function only -- the
    scenario YAML loader (`rules/*.yaml`) is out of scope for M1
    (rules/README.md).

    Tickers in `weights` absent from `shocks` are treated as unshocked, with
    a warning listing them, so a caller notices incomplete scenario coverage
    rather than silently assuming full coverage.
    """
    assert_no_nan(weights, context="apply_stress_scenario")
    unshocked = [ticker for ticker in weights.index if ticker not in shocks]
    warnings = [f"no shock provided for: {unshocked}"] if unshocked else []

    value = float(sum(weight * shocks.get(str(ticker), 0.0) for ticker, weight in weights.items()))
    assert_finite(value, context="apply_stress_scenario")

    return ScalarResult(
        method="linear_shock", sample_size=len(weights), value=value, warnings=warnings
    )
