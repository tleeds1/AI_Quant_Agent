from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from redis.asyncio import Redis

from quantagent.config import settings
from quantagent.data.cache import CacheClient


@pytest.fixture
async def cache_client() -> AsyncIterator[CacheClient]:
    redis: Redis = Redis.from_url(settings.redis_url)
    client = CacheClient(redis)
    yield client
    await redis.flushdb()
    await client.close()


async def test_set_applies_the_requested_ttl(cache_client: CacheClient) -> None:
    await cache_client.set("test:ttl", b"payload", ttl_s=2)

    ttl = await cache_client._redis.ttl("test:ttl")

    assert 0 < ttl <= 2


async def test_get_returns_none_after_delete(cache_client: CacheClient) -> None:
    await cache_client.set("test:delete", b"payload", ttl_s=60)
    await cache_client._redis.delete("test:delete")

    result = await cache_client.get("test:delete")

    assert result is None


async def test_get_round_trips_the_stored_value(cache_client: CacheClient) -> None:
    await cache_client.set("test:roundtrip", b"hello world", ttl_s=60)

    result = await cache_client.get("test:roundtrip")

    assert result == b"hello world"
