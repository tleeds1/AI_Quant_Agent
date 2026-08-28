from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from quantagent.contracts.errors import ToolValidationError
from quantagent.contracts.metrics import MetricValue
from quantagent.contracts.tools import ComputeExpressionInput
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_expression(expr: str, refs: dict[str, float]) -> float:
    """AST-based whitelist evaluator (architecture.md §3.1). Grammar: numeric
    literals, names present in `refs`, `+ - * / ** ()`, unary +/-. Never
    `eval`/`exec`. Raises `ToolValidationError` on any disallowed syntax, an
    unknown name, or division by zero.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolValidationError(f"expr is not valid syntax: {expr!r}") from exc
    try:
        return _eval_node(tree.body, refs)
    except ZeroDivisionError as exc:
        raise ToolValidationError(f"division by zero in expr: {expr!r}") from exc


def _eval_node(node: ast.expr, refs: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolValidationError(f"only numeric literals are allowed, got {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in refs:
            raise ToolValidationError(f"unknown reference {node.id!r} (not present in refs)")
        return float(refs[node.id])
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ToolValidationError(f"operator {type(node.op).__name__} is not allowed")
        return op(_eval_node(node.left, refs), _eval_node(node.right, refs))
    if isinstance(node, ast.UnaryOp):
        unary_op = _UNARY_OPS.get(type(node.op))
        if unary_op is None:
            raise ToolValidationError(f"unary operator {type(node.op).__name__} is not allowed")
        return unary_op(_eval_node(node.operand, refs))
    raise ToolValidationError(f"expression node {type(node).__name__} is not allowed")


@registry.tool(
    name="compute_expression",
    description=(
        "Evaluate a whitelisted arithmetic expression (+ - * / ** and parentheses) over "
        "named numeric values supplied in `refs`. Use for a derived ratio, difference, or "
        "percentage change built from values already present in the ledger, e.g. "
        "expr='a / b - 1', refs={'a': 1.05, 'b': 1.00}. Do NOT use to fetch new data or to "
        "recompute something another tool already provides -- it never calls data/ or quant/, "
        "it only does arithmetic on numbers the caller already has."
    ),
    p95_latency_ms=5,
    est_cost_usd=0.0,
    cache_ttl_s=0,  # pure function of caller-supplied refs; not worth a cache round-trip
    side_effects="READ_ONLY",
)
async def compute_expression(inp: ComputeExpressionInput, ctx: ToolContext) -> MetricValue:
    value = evaluate_expression(inp.expr, inp.refs)
    return ctx.wrap_metric(
        metric_id=f"expr:{inp.expr}",
        value=value,
        unit=inp.unit,
        method="compute_expression",
        data_sources=["ledger"],
    )
