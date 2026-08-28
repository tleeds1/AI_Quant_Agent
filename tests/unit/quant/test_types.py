from __future__ import annotations

import dataclasses

import pytest

from quantagent.quant.types import ScalarResult


def test_result_dataclasses_are_frozen() -> None:
    result = ScalarResult(method="test", sample_size=10, value=1.0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.value = 2.0  # type: ignore[misc]


def test_warnings_default_to_empty_list_not_shared_mutable_default() -> None:
    first = ScalarResult(method="test", sample_size=10, value=1.0)
    second = ScalarResult(method="test", sample_size=10, value=2.0)

    first.warnings.append("only on first")

    assert first.warnings == ["only on first"]
    assert second.warnings == []
