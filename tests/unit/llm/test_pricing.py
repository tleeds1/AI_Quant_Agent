from __future__ import annotations

from quantagent.llm.pricing import estimate_cost_usd


def test_known_model_returns_nonzero_cost() -> None:
    cost = estimate_cost_usd("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 18.00


def test_unknown_model_returns_zero_and_does_not_raise() -> None:
    cost = estimate_cost_usd("some-future-model", input_tokens=1000, output_tokens=1000)
    assert cost == 0.0
