from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from quantagent.data.providers.embeddings import EMBEDDING_DIM

TransactionSide = Literal["buy", "sell"]
TRANSACTION_SIDES: tuple[TransactionSide, ...] = ("buy", "sell")

# Kept in sync with TRANSACTION_SIDES by construction: a drifted CHECK
# constraint would let a side the Python layer cannot represent into the table.
_SIDE_CHECK_SQL = "side IN (" + ", ".join(f"'{side}'" for side in TRANSACTION_SIDES) + ")"


class Base(DeclarativeBase):
    pass


class Portfolio(Base):
    """A tenant-scoped portfolio: mandate, base currency, benchmark. Minimal
    fields needed to seed and analyse a demo portfolio (M1) -- the full
    mandate/constraints engine is out of scope; `mandate_constraints` is a
    free-form, unvalidated JSON placeholder.
    """

    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    base_currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    benchmark_ticker: Mapped[str] = mapped_column(String(16), nullable=False, default="SPY")
    mandate_constraints: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    holdings: Mapped[list[Holding]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    transactions: Mapped[list[Transaction]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_portfolios_tenant_id_id", "tenant_id", "id"),)


class Holding(Base):
    """A single position in a `Portfolio` as of a given date.

    No `tenant_id` column here -- tenancy flows entirely through the
    mandatory join to `Portfolio.tenant_id`, so there is one source of
    truth for tenant scope, never two columns that could drift apart.
    """

    __tablename__ = "holdings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    cost_basis: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    as_of: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")

    __table_args__ = (
        UniqueConstraint(
            "portfolio_id", "ticker", "as_of", name="uq_holdings_portfolio_ticker_as_of"
        ),
        Index("ix_holdings_portfolio_id_ticker", "portfolio_id", "ticker"),
    )


class Transaction(Base):
    """A single executed trade against a `Portfolio`.

    No `tenant_id` column here -- tenancy flows entirely through the
    mandatory join to `Portfolio.tenant_id`, so there is one source of
    truth for tenant scope, never two columns that could drift apart.

    Unlike `Holding` there is deliberately no uniqueness constraint: the
    same ticker can legitimately be bought twice on one day at two prices,
    so there is no natural key to upsert on and writes replace a whole
    portfolio's history rather than merging row by row.
    """

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    portfolio: Mapped[Portfolio] = relationship(back_populates="transactions")

    __table_args__ = (
        CheckConstraint(_SIDE_CHECK_SQL, name="ck_transactions_side"),
        Index("ix_transactions_portfolio_id_trade_date", "portfolio_id", "trade_date"),
    )


class Filing(Base):
    """A single SEC EDGAR filing (10-K/10-Q/8-K/...): public, tenant-agnostic.

    No `tenant_id` column, and `FilingsRepository` does not inherit
    `RepositoryBase` -- every SEC filing is identical for every tenant, so
    there is no tenant-owning parent to join through (unlike `Holding`/
    `Transaction`'s join to `Portfolio.tenant_id`). See `FilingsRepository`'s
    own docstring for the full reasoning.
    """

    __tablename__ = "filings"

    id: Mapped[str] = mapped_column(String(25), primary_key=True)  # == accession_no
    cik: Mapped[str] = mapped_column(String(10), nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    company_name: Mapped[str] = mapped_column(String(300), nullable=False)
    form_type: Mapped[str] = mapped_column(String(10), nullable=False)
    filed_at: Mapped[date] = mapped_column(Date, nullable=False)
    period_of_report: Mapped[date | None] = mapped_column(Date, nullable=True)
    primary_document_url: Mapped[str] = mapped_column(String(500), nullable=False)
    source_tier: Mapped[str] = mapped_column(String(2), nullable=False, default="T1")
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunks: Mapped[list[FilingChunk]] = relationship(
        back_populates="filing", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_filings_ticker_form_type", "ticker", "form_type", "filed_at"),)


class FilingChunk(Base):
    """One item-scoped, chunked passage of a `Filing`, with an optional
    embedding vector.

    `embedding` is nullable: rows are written at ingest time and vectors
    backfilled independently, so a future embedding-model swap can
    recompute vectors without re-chunking.

    No `content_tsv` column here. The BM25 full-text index is a Postgres
    *generated* column created directly by the Alembic migration (raw DDL),
    deliberately NOT mapped as a SQLAlchemy attribute: mapping a
    `to_tsvector(...)`-computed column here would put that Postgres-only
    expression into `Base.metadata`, which every existing repository unit
    test's `Base.metadata.create_all()` against sqlite would then try (and
    fail) to compile -- sqlite has no `to_tsvector` function.
    `FilingsRepository.search_bm25` reads that column through raw SQL
    instead, and like `search_dense` (pgvector's `<=>` operator, also
    unsupported by sqlite) is exercised only by the Postgres integration
    tests, never the sqlite unit-test fixture.
    """

    __tablename__ = "filing_chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filing_id: Mapped[str] = mapped_column(
        String(25), ForeignKey("filings.id", ondelete="CASCADE"), nullable=False
    )
    item: Mapped[str] = mapped_column(String(8), nullable=False)
    section_path: Mapped[str] = mapped_column(String(200), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    filing: Mapped[Filing] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "filing_id", "item", "chunk_index", name="uq_filing_chunks_filing_item_index"
        ),
        Index("ix_filing_chunks_filing_id", "filing_id"),
    )


class Trace(Base):
    """Persisted trace of a user query execution (architecture.md §9.1).
    Contains metadata and JSON dumps of intake, planner, ledger, verifications,
    and LLM calls to allow full auditability and nightly reproducibility replay.
    """

    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    portfolio_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    intent_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    intent_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    intent_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ledger: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    answer: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    verification_report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    guardrail_decisions: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    llm_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_traces_tenant_id_created_at", "tenant_id", "created_at"),)


class AuditLogEntry(Base):
    """Tamper-evident, hash-chained audit log of released answers (architecture.md §9.5)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    data_sources: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    recommendation: Mapped[str] = mapped_column(String(50), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    verifier_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    released_by: Mapped[str] = mapped_column(String(32), nullable=False)
    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_audit_log_tenant_id_created_at", "tenant_id", "created_at"),)
