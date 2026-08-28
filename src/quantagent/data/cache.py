from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

from redis.asyncio import Redis

from quantagent.config import settings


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot canonicalize value of type {type(value)!r} for hashing")


def compute_inputs_hash(**kwargs: object) -> str:
    """Sha256 of the sorted-key JSON encoding of `kwargs`.

    Mirrors `contracts.Provenance.inputs_hash`'s documented semantics
    (cache key + replay key) so `tools/` can reuse this exact function in
    M2 rather than reimplementing hash logic.
    """
    canonical = json.dumps(kwargs, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class CacheClient:
    """Thin async wrapper around Redis: generic bytes in, bytes out.

    Knows nothing about pandas or prices -- domain-specific serialization
    and TTL policy live with the caller (e.g. `data/providers/prices.py`).
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @classmethod
    def from_settings(cls) -> CacheClient:
        return cls(Redis.from_url(settings.redis_url))

    async def get(self, key: str) -> bytes | None:
        # decode_responses defaults to False, so this client is bytes-mode;
        # the str branch below only matters if that default is ever changed.
        value = await self._redis.get(key)
        if value is None or isinstance(value, bytes):
            return value
        return value.encode()

    async def set(self, key: str, value: bytes, ttl_s: int) -> None:
        await self._redis.set(key, value, ex=ttl_s)

    async def close(self) -> None:
        await self._redis.aclose()
