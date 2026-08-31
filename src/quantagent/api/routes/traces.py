from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException

from quantagent.api.deps import AppResources, get_app_resources
from quantagent.data.repositories.trace_repository import TraceRepository

router = APIRouter(prefix="/v1", tags=["traces"])


@router.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    resources: AppResources = Depends(get_app_resources),
) -> dict[str, Any]:
    """Load and return a persisted trace by ID, validating tenant scope (I9)."""
    if x_tenant_id is None or not x_tenant_id.strip():
        raise HTTPException(status_code=400, detail="X-Tenant-Id header is required")

    repo = TraceRepository(resources.session_factory)
    trace_data = await repo.get_trace(trace_id, tenant_id=x_tenant_id)
    if trace_data is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found for this tenant")

    return trace_data
