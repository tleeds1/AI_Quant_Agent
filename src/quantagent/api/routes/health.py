from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/healthz")
async def get_health() -> dict[str, str]:
    """Liveness probe. Returns 200 with a static payload; no dependency checks yet."""
    return {"status": "ok"}
