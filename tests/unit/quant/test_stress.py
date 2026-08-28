from __future__ import annotations

import pandas as pd
import pytest

from quantagent.quant.stress import apply_stress_scenario


def test_apply_stress_scenario_matches_manual_weighted_shock_sum() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4})
    shocks = {"A": -0.10, "B": -0.20}

    result = apply_stress_scenario(weights, shocks)

    assert result.value == pytest.approx(0.6 * -0.10 + 0.4 * -0.20)


def test_unshocked_tickers_produce_warning_listing_them() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    shocks = {"A": -0.10}

    result = apply_stress_scenario(weights, shocks)

    assert any("B" in w for w in result.warnings)


def test_full_coverage_scenario_produces_no_warnings() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    shocks = {"A": -0.10, "B": -0.05}

    result = apply_stress_scenario(weights, shocks)

    assert result.warnings == []
