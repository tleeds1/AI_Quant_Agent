from __future__ import annotations

import hashlib
from typing import Any

from sqlalchemy import select

from quantagent.data.models import AuditLogEntry as AuditLogEntryRow
from quantagent.data.models import Trace as TraceRow
from quantagent.data.repositories.base import RepositoryBase


class TraceRepository(RepositoryBase):
    """Tenant-scoped database access for Trace persistence and AuditLogEntry chaining."""

    async def save_trace(self, trace_id: str, tenant_id: str, **kwargs: Any) -> None:
        self._require_tenant(tenant_id)
        async with self._session_factory() as session:
            existing = await session.get(TraceRow, trace_id)
            if existing is not None:
                # Update existing trace fields
                existing.tenant_id = tenant_id
                for k, v in kwargs.items():
                    setattr(existing, k, v)
            else:
                # Insert a new trace
                new_trace = TraceRow(id=trace_id, tenant_id=tenant_id, **kwargs)
                session.add(new_trace)
            await session.commit()

    async def get_trace(self, trace_id: str, tenant_id: str) -> dict[str, Any] | None:
        self._require_tenant(tenant_id)
        async with self._session_factory() as session:
            stmt = select(TraceRow).where(TraceRow.id == trace_id, TraceRow.tenant_id == tenant_id)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            return {
                "id": row.id,
                "tenant_id": row.tenant_id,
                "question": row.question,
                "portfolio_id": row.portfolio_id,
                "intent_label": row.intent_label,
                "intent_confidence": (
                    float(row.intent_confidence) if row.intent_confidence is not None else None
                ),
                "intent_rationale": row.intent_rationale,
                "plan": row.plan,
                "ledger": row.ledger,
                "answer": row.answer,
                "verification_report": row.verification_report,
                "guardrail_decisions": row.guardrail_decisions,
                "llm_calls": row.llm_calls,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }

    async def save_audit_log(
        self,
        *,
        trace_id: str,
        tenant_id: str,
        question: str,
        data_sources: dict[str, Any] | None,
        recommendation: str,
        summary: str,
        verifier_verdict: str,
        released_by: str,
    ) -> str:
        self._require_tenant(tenant_id)
        async with self._session_factory() as session:
            # Query for the latest hash to chain on
            stmt = select(AuditLogEntryRow.hash).order_by(AuditLogEntryRow.id.desc()).limit(1)
            previous_hash = (await session.execute(stmt)).scalar_one_or_none() or "0" * 64

            # Compute tamper-evident SHA-256 hash
            payload = (
                f"{previous_hash}|{tenant_id}|{trace_id}|{question}|"
                f"{verifier_verdict}|{released_by}|{recommendation}"
            )
            current_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

            new_entry = AuditLogEntryRow(
                trace_id=trace_id,
                tenant_id=tenant_id,
                question=question,
                data_sources=data_sources,
                recommendation=recommendation,
                summary=summary,
                verifier_verdict=verifier_verdict,
                released_by=released_by,
                previous_hash=previous_hash,
                hash=current_hash,
            )
            session.add(new_entry)
            await session.commit()
            return current_hash
