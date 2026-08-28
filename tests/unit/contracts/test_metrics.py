from __future__ import annotations

from quantagent.contracts.metrics import MetricValue
from tests.unit.contracts.test_provenance import build_provenance


def build_metric_value(**overrides: object) -> MetricValue:
    defaults: dict[str, object] = {
        "metric_id": "portfolio_var_95_1d",
        "value": 0.0243,
        "unit": "ratio",
        "method": "historical_simulation",
        "window": "504d",
        "ci_95": None,
        "provenance": build_provenance(),
    }
    defaults.update(overrides)
    return MetricValue(**defaults)


def test_metric_value_round_trips_through_json_with_nested_provenance() -> None:
    original = build_metric_value()

    restored = MetricValue.model_validate_json(original.model_dump_json())

    assert restored == original
