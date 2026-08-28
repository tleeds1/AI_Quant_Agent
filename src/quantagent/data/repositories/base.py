from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quantagent.contracts.errors import DataError


class RepositoryBase:
    """Structural I9 enforcement point: every public repository method must
    call `_require_tenant` before touching the database, instead of each
    method hand-rolling its own check.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _require_tenant(tenant_id: str) -> None:
        if not tenant_id or not tenant_id.strip():
            raise DataError("tenant_id is required for repository access (I9)")
