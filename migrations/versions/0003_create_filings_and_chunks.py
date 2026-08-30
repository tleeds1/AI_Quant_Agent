"""create filings and filing_chunks

Revision ID: f448e6649fcf
Revises: 96fab3fd55cf
Create Date: 2026-08-29 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "f448e6649fcf"
down_revision: Union[str, Sequence[str], None] = "96fab3fd55cf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Mirrors data/providers/embeddings.py::EMBEDDING_DIM -- not imported directly
# so this migration keeps reading correctly even if that module's import
# path ever changes; kept in sync by the same PR that changes either.
_EMBEDDING_DIM = 384


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "filings",
        sa.Column("id", sa.String(length=25), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("company_name", sa.String(length=300), nullable=False),
        sa.Column("form_type", sa.String(length=10), nullable=False),
        sa.Column("filed_at", sa.Date(), nullable=False),
        sa.Column("period_of_report", sa.Date(), nullable=True),
        sa.Column("primary_document_url", sa.String(length=500), nullable=False),
        sa.Column("source_tier", sa.String(length=2), nullable=False, server_default="T1"),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_filings_ticker_form_type", "filings", ["ticker", "form_type", "filed_at"], unique=False
    )

    op.create_table(
        "filing_chunks",
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("filing_id", sa.String(length=25), nullable=False),
        sa.Column("item", sa.String(length=8), nullable=False),
        sa.Column("section_path", sa.String(length=200), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=True),
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["filing_id"], ["filings.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("chunk_id"),
        sa.UniqueConstraint(
            "filing_id", "item", "chunk_index", name="uq_filing_chunks_filing_item_index"
        ),
    )
    op.create_index("ix_filing_chunks_filing_id", "filing_chunks", ["filing_id"], unique=False)

    # BM25 full-text search column: a Postgres *generated* column, created
    # here via raw DDL rather than op.create_table's column list, and
    # deliberately NOT mapped in data/models.py::FilingChunk (see that
    # class's docstring) -- sqlite, used by the unit-test fixture, has no
    # `to_tsvector` function and would fail to compile it.
    op.execute(
        "ALTER TABLE filing_chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED"
    )
    op.execute("CREATE INDEX ix_filing_chunks_content_tsv ON filing_chunks USING gin (content_tsv)")
    op.execute(
        "CREATE INDEX ix_filing_chunks_embedding_hnsw ON filing_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_filing_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_filing_chunks_content_tsv")
    op.drop_index("ix_filing_chunks_filing_id", table_name="filing_chunks")
    op.drop_table("filing_chunks")
    op.drop_index("ix_filings_ticker_form_type", table_name="filings")
    op.drop_table("filings")
