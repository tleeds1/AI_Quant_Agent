from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest

from quantagent.data.cache import CacheClient, compute_inputs_hash


def test_compute_inputs_hash_is_deterministic() -> None:
    first = compute_inputs_hash(tickers=["AAPL", "MSFT"], alpha=0.95)
    second = compute_inputs_hash(tickers=["AAPL", "MSFT"], alpha=0.95)

    assert first == second


def test_compute_inputs_hash_is_kwarg_order_independent() -> None:
    first = compute_inputs_hash(a=1, b=2)
    second = compute_inputs_hash(b=2, a=1)

    assert first == second


def test_compute_inputs_hash_differs_on_different_values() -> None:
    first = compute_inputs_hash(alpha=0.95)
    second = compute_inputs_hash(alpha=0.99)

    assert first != second


def test_compute_inputs_hash_handles_dates() -> None:
    result = compute_inputs_hash(as_of=date(2026, 8, 22))

    assert isinstance(result, str)
    assert len(result) == 64


@pytest.mark.asyncio
async def test_cache_client_get_returns_none_on_miss() -> None:
    redis = AsyncMock()
    redis.get.return_value = None
    client = CacheClient(redis)

    result = await client.get("some-key")

    assert result is None
    redis.get.assert_awaited_once_with("some-key")


@pytest.mark.asyncio
async def test_cache_client_set_passes_ttl_to_redis() -> None:
    redis = AsyncMock()
    client = CacheClient(redis)

    await client.set("some-key", b"payload", ttl_s=900)

    redis.set.assert_awaited_once_with("some-key", b"payload", ex=900)


@pytest.mark.asyncio
async def test_cache_client_close_closes_underlying_redis_connection() -> None:
    redis = AsyncMock()
    client = CacheClient(redis)

    await client.close()

    redis.aclose.assert_awaited_once()
