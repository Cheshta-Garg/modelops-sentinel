"""Add delayed ground-truth performance columns to perf_metrics.

Revision ID: 0002_add_ground_truth_perf_metrics
Revises: 0001_initial_schema
Create Date: 2026-11-18 13:25:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_add_ground_truth_perf_metrics"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "perf_metrics",
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("perf_metrics", sa.Column("accuracy", sa.Float(), nullable=True))
    op.add_column("perf_metrics", sa.Column("precision", sa.Float(), nullable=True))
    op.add_column("perf_metrics", sa.Column("recall", sa.Float(), nullable=True))
    op.add_column("perf_metrics", sa.Column("f1_score", sa.Float(), nullable=True))
    op.add_column(
        "perf_metrics",
        sa.Column("confusion_matrix", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "perf_metrics",
        sa.Column(
            "evaluated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_perf_metrics_evaluated_at", "perf_metrics", ["evaluated_at"])

    # Relax legacy operational columns so delayed ground-truth rows do not mix operational telemetry
    op.alter_column("perf_metrics", "total_requests", existing_type=sa.Integer(), nullable=True)
    op.alter_column("perf_metrics", "cache_hits", existing_type=sa.Integer(), nullable=True)
    op.alter_column("perf_metrics", "p50_latency_ms", existing_type=sa.Float(), nullable=True)
    op.alter_column("perf_metrics", "p95_latency_ms", existing_type=sa.Float(), nullable=True)
    op.alter_column("perf_metrics", "p99_latency_ms", existing_type=sa.Float(), nullable=True)
    op.alter_column("perf_metrics", "error_count", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    op.alter_column("perf_metrics", "error_count", existing_type=sa.Integer(), nullable=False)
    op.alter_column("perf_metrics", "p99_latency_ms", existing_type=sa.Float(), nullable=False)
    op.alter_column("perf_metrics", "p95_latency_ms", existing_type=sa.Float(), nullable=False)
    op.alter_column("perf_metrics", "p50_latency_ms", existing_type=sa.Float(), nullable=False)
    op.alter_column("perf_metrics", "cache_hits", existing_type=sa.Integer(), nullable=False)
    op.alter_column("perf_metrics", "total_requests", existing_type=sa.Integer(), nullable=False)

    op.drop_index("ix_perf_metrics_evaluated_at", table_name="perf_metrics")
    op.drop_column("perf_metrics", "evaluated_at")
    op.drop_column("perf_metrics", "confusion_matrix")
    op.drop_column("perf_metrics", "f1_score")
    op.drop_column("perf_metrics", "recall")
    op.drop_column("perf_metrics", "precision")
    op.drop_column("perf_metrics", "accuracy")
    op.drop_column("perf_metrics", "sample_count")
