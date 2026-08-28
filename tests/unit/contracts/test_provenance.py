from __future__ import annotations

from datetime import date, datetime

from quantagent.contracts.provenance import Provenance


def build_provenance(**overrides: object) -> Provenance:
    defaults: dict[str, object] = {
        "tool_call_id": "tc_01",
        "tool_name": "calculate_portfolio_var",
        "as_of": date(2026, 8, 22),
        "computed_at": datetime(2026, 8, 22, 21, 5, 0),
        "inputs_hash": "a" * 64,
        "data_sources": ["yfinance:adjusted_close"],
        "estimator": "historical_simulation",
        "sample_size": 504,
        "seed": None,
        "warnings": [],
    }
    defaults.update(overrides)
    return Provenance(**defaults)


def test_provenance_round_trips_through_json() -> None:
    original = build_provenance()

    restored = Provenance.model_validate_json(original.model_dump_json())

    assert restored == original


def test_provenance_warnings_default_to_empty_list() -> None:
    provenance = Provenance(
        tool_call_id="tc_02",
        tool_name="get_prices",
        as_of=date(2026, 8, 22),
        computed_at=datetime(2026, 8, 22, 21, 5, 0),
        inputs_hash="b" * 64,
        data_sources=["yfinance:adjusted_close"],
    )

    assert provenance.warnings == []
