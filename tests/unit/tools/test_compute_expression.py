from __future__ import annotations

import pytest

from quantagent.contracts.errors import ToolValidationError
from quantagent.contracts.tools import ComputeExpressionInput
from quantagent.tools.compute_expression import compute_expression, evaluate_expression
from quantagent.tools.registry import registry
from tests.unit.tools.builders import build_tool_context


def test_addition() -> None:
    assert evaluate_expression("a + b", {"a": 1.0, "b": 2.0}) == pytest.approx(3.0)


def test_subtraction() -> None:
    assert evaluate_expression("a - b", {"a": 5.0, "b": 2.0}) == pytest.approx(3.0)


def test_multiplication() -> None:
    assert evaluate_expression("a * b", {"a": 3.0, "b": 4.0}) == pytest.approx(12.0)


def test_division() -> None:
    assert evaluate_expression("a / b", {"a": 10.0, "b": 4.0}) == pytest.approx(2.5)


def test_power() -> None:
    assert evaluate_expression("2 ** 10", {}) == pytest.approx(1024.0)


def test_unary_minus() -> None:
    assert evaluate_expression("-a", {"a": 5.0}) == pytest.approx(-5.0)


def test_unary_plus() -> None:
    assert evaluate_expression("+a", {"a": 5.0}) == pytest.approx(5.0)


def test_nested_parentheses_and_operator_precedence() -> None:
    result = evaluate_expression("(a + b) / (c - 1)", {"a": 1.0, "b": 3.0, "c": 3.0})
    assert result == pytest.approx(2.0)


def test_derived_ratio_expression() -> None:
    result = evaluate_expression("a / b - 1", {"a": 1.05, "b": 1.00})
    assert result == pytest.approx(0.05)


def test_bool_literal_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        evaluate_expression("1 + True", {})


def test_string_literal_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        evaluate_expression("'not a number'", {})


def test_function_call_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        evaluate_expression("__import__('os').system('echo hi')", {})


def test_attribute_access_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        evaluate_expression("a.bit_length", {"a": 1.0})


def test_comparison_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        evaluate_expression("a < b", {"a": 1.0, "b": 2.0})


def test_unknown_name_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        evaluate_expression("unknown_ref", {})


def test_division_by_zero_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        evaluate_expression("a / b", {"a": 1.0, "b": 0.0})


def test_invalid_syntax_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        evaluate_expression("a +* b", {"a": 1.0, "b": 2.0})


def test_list_literal_is_rejected() -> None:
    with pytest.raises(ToolValidationError):
        evaluate_expression("[1, 2, 3]", {})


async def test_compute_expression_tool_wraps_result_with_provenance() -> None:
    ctx = build_tool_context().for_call(tool_name="compute_expression", inputs_hash="h")

    result = await compute_expression(
        ComputeExpressionInput(expr="a / b - 1", refs={"a": 1.05, "b": 1.00}), ctx
    )

    assert result.value == pytest.approx(0.05)
    assert result.provenance.data_sources == ["ledger"]


async def test_compute_expression_is_registered_and_invocable_via_registry() -> None:
    ctx = build_tool_context()

    result = await registry.invoke(
        "compute_expression", {"expr": "a * 2", "refs": {"a": 21.0}}, ctx
    )

    assert result.value == pytest.approx(42.0)
