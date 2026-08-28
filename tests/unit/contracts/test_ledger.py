from __future__ import annotations

from quantagent.contracts.ledger import Ledger, ToolCallRecord


def build_tool_call_record(**overrides: object) -> ToolCallRecord:
    defaults: dict[str, object] = {
        "call_id": "tc_05",
        "tool_name": "calculate_portfolio_var",
        "args": {"portfolio_id": "p_1", "alpha": 0.95},
        "args_hash": "c" * 64,
        "status": "OK",
        "latency_ms": 220,
        "cost_usd": 0.0,
        "result": {"var_95": 0.0243},
        "error": None,
    }
    defaults.update(overrides)
    return ToolCallRecord(**defaults)


def build_ledger(**overrides: object) -> Ledger:
    defaults: dict[str, object] = {
        "trace_id": "tr_9f3c1a",
        "calls": [build_tool_call_record()],
        "numeric_index": {"tc_05.result.var_95": 0.0243},
    }
    defaults.update(overrides)
    return Ledger(**defaults)


def test_tool_call_record_round_trips_through_json() -> None:
    original = build_tool_call_record()

    restored = ToolCallRecord.model_validate_json(original.model_dump_json())

    assert restored == original


def test_ledger_round_trips_through_json_with_nested_calls() -> None:
    original = build_ledger()

    restored = Ledger.model_validate_json(original.model_dump_json())

    assert restored == original
