"""create traces and audit_log

Revision ID: a67b8e90c12f
Revises: f448e6649fcf
Create Date: 2026-08-30 15:28:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a67b8e90c12f"
down_revision: Union[str, Sequence[str], None] = "f448e6649fcf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "traces",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("portfolio_id", sa.String(length=64), nullable=True),
        sa.Column("intent_label", sa.String(length=32), nullable=True),
        sa.Column("intent_confidence", sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column("intent_rationale", sa.Text(), nullable=True),
        sa.Column("plan", sa.JSON(), nullable=True),
        sa.Column("ledger", sa.JSON(), nullable=True),
        sa.Column("answer", sa.JSON(), nullable=True),
        sa.Column("verification_report", sa.JSON(), nullable=True),
        sa.Column("guardrail_decisions", sa.JSON(), nullable=True),
        sa.Column("llm_calls", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_traces_tenant_id_created_at",
        "traces",
        ["tenant_id", "created_at"],
        unique=False,
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("data_sources", sa.JSON(), nullable=True),
        sa.Column("recommendation", sa.String(length=50), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("verifier_verdict", sa.String(length=32), nullable=False),
        sa.Column("released_by", sa.String(length=32), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=False),
        sa.Column("hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_audit_log_tenant_id_created_at",
        "audit_log",
        ["tenant_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_audit_log_trace_id",
        "audit_log",
        ["trace_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_audit_log_trace_id", table_name="audit_log")
    op.drop_index("ix_audit_log_tenant_id_created_at", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_traces_tenant_id_created_at", table_name="traces")
    op.drop_table("traces")
