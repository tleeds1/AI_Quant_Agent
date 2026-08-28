from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.quant.validation import (
    as_float64,
    assert_aligned_index,
    assert_finite,
    assert_no_nan,
    assert_single_currency,
    assert_weights_sum_to_one,
)


def test_as_float64_coerces_series_to_float64_array() -> None:
    series = pd.Series([1, 2, 3], dtype="int32")

    result = as_float64(series)

    assert result.dtype == np.float64


def test_as_float64_coerces_raw_array_to_float64() -> None:
    result = as_float64(np.array([1, 2, 3], dtype=np.int32))

    assert result.dtype == np.float64


def test_assert_finite_raises_on_nan() -> None:
    with pytest.raises(ValueError):
        assert_finite(float("nan"), context="test")


def test_assert_finite_raises_on_infinite() -> None:
    with pytest.raises(ValueError):
        assert_finite(float("inf"), context="test")


def test_assert_finite_passes_on_finite_value() -> None:
    assert_finite(1.23, context="test")


def test_assert_no_nan_raises_when_nan_present() -> None:
    series = pd.Series([1.0, float("nan"), 3.0])

    with pytest.raises(ValueError):
        assert_no_nan(series, context="test")


def test_assert_no_nan_passes_when_clean() -> None:
    series = pd.Series([1.0, 2.0, 3.0])

    assert_no_nan(series, context="test")


def test_assert_weights_sum_to_one_raises_outside_tolerance() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.3})

    with pytest.raises(ValueError):
        assert_weights_sum_to_one(weights)


def test_assert_weights_sum_to_one_passes_within_tolerance() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})

    assert_weights_sum_to_one(weights)


def test_assert_aligned_index_raises_on_mismatch() -> None:
    a = pd.Series([1.0, 2.0], index=pd.bdate_range("2020-01-01", periods=2))
    b = pd.Series([1.0, 2.0], index=pd.bdate_range("2020-02-01", periods=2))

    with pytest.raises(ValueError):
        assert_aligned_index(a, b, context="test")


def test_assert_single_currency_raises_on_mixed_input() -> None:
    with pytest.raises(ValueError):
        assert_single_currency(["USD", "EUR"])


def test_assert_single_currency_passes_on_uniform_input() -> None:
    assert_single_currency(["USD", "USD", "USD"])
