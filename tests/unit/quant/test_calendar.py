from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.quant.calendar import align_calendars


def _panel(n: int = 20) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "AAA": np.linspace(100.0, 119.0, n),
            "BBB": np.linspace(50.0, 69.0, n),
        },
        index=index,
    )


def test_align_calendars_forward_fills_gap_within_window() -> None:
    panel = _panel()
    panel.iloc[5:7, panel.columns.get_loc("BBB")] = np.nan  # 2-day gap, within default window of 5

    aligned, warnings = align_calendars(panel)

    assert not aligned.isna().any().any()
    assert len(aligned) == len(panel)
    assert any("BBB" in w for w in warnings)


def test_align_calendars_drops_rows_with_gap_beyond_window() -> None:
    panel = _panel()
    panel.iloc[5:14, panel.columns.get_loc("BBB")] = np.nan  # 9-day gap, beyond default window of 5

    aligned, warnings = align_calendars(panel)

    assert not aligned.isna().any().any()
    assert len(aligned) < len(panel)
    assert any("dropped" in w for w in warnings)


def test_align_calendars_fills_gap_exactly_at_boundary() -> None:
    panel = _panel()
    start = 5
    panel.iloc[start : start + 5, panel.columns.get_loc("BBB")] = np.nan  # exactly 5-day gap

    aligned, _warnings = align_calendars(panel, max_forward_fill_days=5)

    assert not aligned.isna().any().any()
    assert len(aligned) == len(panel)


def test_align_calendars_returns_no_warnings_when_no_gaps() -> None:
    panel = _panel()

    aligned, warnings = align_calendars(panel)

    assert warnings == []
    assert len(aligned) == len(panel)
