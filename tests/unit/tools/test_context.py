from __future__ import annotations

from datetime import date

import pytest

from tests.unit.tools.builders import build_tool_context


def test_for_call_shares_resource_references_across_calls() -> None:
    ctx = build_tool_context()

    call_a = ctx.for_call(tool_name="tool_a", inputs_hash="hash_a")

    assert call_a.portfolios is ctx.portfolios
    assert call_a.prices is ctx.prices
    assert call_a.fundamentals is ctx.fundamentals
    assert call_a.factors is ctx.factors
    assert call_a.cache is ctx.cache


def test_for_call_generates_distinct_tool_call_ids_when_not_given_one() -> None:
    ctx = build_tool_context()

    call_a = ctx.for_call(tool_name="tool_a", inputs_hash="hash_a")
    call_b = ctx.for_call(tool_name="tool_b", inputs_hash="hash_b")

    provenance_a = call_a.build_provenance()
    provenance_b = call_b.build_provenance()
    assert provenance_a.tool_call_id != provenance_b.tool_call_id


def test_for_call_honors_an_explicit_tool_call_id() -> None:
    ctx = build_tool_context()

    bound = ctx.for_call(tool_name="tool_a", inputs_hash="hash_a", tool_call_id="tc_explicit")

    assert bound.build_provenance().tool_call_id == "tc_explicit"


def test_build_provenance_raises_on_unbound_context() -> None:
    ctx = build_tool_context()

    with pytest.raises(RuntimeError):
        ctx.build_provenance()


def test_build_provenance_defaults_as_of_to_today() -> None:
    ctx = build_tool_context().for_call(tool_name="tool_a", inputs_hash="hash_a")

    provenance = ctx.build_provenance()

    assert provenance.as_of == date.today()


def test_build_provenance_honors_explicit_as_of() -> None:
    ctx = build_tool_context().for_call(tool_name="tool_a", inputs_hash="hash_a")
    explicit_date = date(2026, 1, 1)

    provenance = ctx.build_provenance(as_of=explicit_date)

    assert provenance.as_of == explicit_date


def test_wrap_metric_builds_a_fully_populated_metric_value() -> None:
    ctx = build_tool_context().for_call(tool_name="tool_a", inputs_hash="hash_a")

    metric = ctx.wrap_metric(
        "some_metric", 0.05, "ratio", "some_method", sample_size=100, data_sources=["fake"]
    )

    assert metric.metric_id == "some_metric"
    assert metric.value == 0.05
    assert metric.unit == "ratio"
    assert metric.method == "some_method"
    assert metric.provenance.tool_name == "tool_a"
    assert metric.provenance.inputs_hash == "hash_a"
    assert metric.provenance.sample_size == 100
    assert metric.provenance.data_sources == ["fake"]
