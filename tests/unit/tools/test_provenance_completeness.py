from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

import quantagent.tools  # noqa: F401 -- import side effect: populates the registry
from quantagent.contracts.provenance import Provenance
from quantagent.tools.registry import registry
from tests.unit.tools.builders import DEFAULT_PORTFOLIO_ID, build_tool_context

_MINIMAL_VALID_ARGS: dict[str, dict[str, Any]] = {
    "get_portfolio": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "get_holdings": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "get_transactions": {
        "portfolio_id": DEFAULT_PORTFOLIO_ID,
        "start": "2020-01-01",
        "end": "2030-01-01",
    },
    "get_prices": {"tickers": ["AAA"], "start": "2024-01-01", "end": "2024-12-31"},
    "get_returns": {"tickers": ["AAA"]},
    "get_fundamentals": {"ticker": "AAA"},
    "get_sector_exposure": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "get_factor_exposure": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "get_correlation_matrix": {"tickers": ["AAA", "BBB"]},
    "get_concentration_metrics": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "calculate_portfolio_var": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "calculate_cvar": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "calculate_component_var": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "calculate_max_drawdown": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "get_portfolio_beta": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "calculate_tracking_error": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
    "compute_expression": {"expr": "a + b", "refs": {"a": 1.0, "b": 2.0}},
    "generate_risk_report": {"portfolio_id": DEFAULT_PORTFOLIO_ID},
}


def _find_provenances(obj: object) -> list[Provenance]:
    """Recursively walk a Pydantic model/list/dict tree and collect every
    `Provenance` found -- the mechanical form of the DoD's "provenance
    completeness asserted for all" requirement.
    """
    if isinstance(obj, Provenance):
        return [obj]
    if isinstance(obj, BaseModel):
        found: list[Provenance] = []
        for field_name in type(obj).model_fields:
            found.extend(_find_provenances(getattr(obj, field_name)))
        return found
    if isinstance(obj, (list, tuple)):
        return [p for item in obj for p in _find_provenances(item)]
    if isinstance(obj, dict):
        return [p for value in obj.values() for p in _find_provenances(value)]
    return []


def test_every_registered_tool_has_a_minimal_args_entry() -> None:
    registered_names = {spec.name for spec in registry.list_tools()}

    assert set(_MINIMAL_VALID_ARGS) == registered_names


@pytest.mark.parametrize("tool_name", sorted(_MINIMAL_VALID_ARGS))
async def test_tool_output_carries_complete_provenance(tool_name: str) -> None:
    ctx = build_tool_context()

    result = await registry.invoke(tool_name, _MINIMAL_VALID_ARGS[tool_name], ctx)

    provenances = _find_provenances(result)
    assert len(provenances) >= 1
    for provenance in provenances:
        assert provenance.tool_call_id
        assert provenance.tool_name
        assert provenance.inputs_hash
        assert len(provenance.data_sources) >= 1
