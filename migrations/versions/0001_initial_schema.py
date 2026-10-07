"""Initial schema migration: creates all 12 platform tables.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-10-05 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. users
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role", "users", ["role"])

    # 2. models
    op.create_table(
        "models",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("task_type", sa.String(length=50), nullable=False, server_default="binary_classification"),
        sa.Column("created_by_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_models_name", "models", ["name"], unique=True)
    op.create_index("ix_models_created_by_id", "models", ["created_by_id"])

    # 3. model_versions
    op.create_table(
        "model_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_id", sa.UUID(), nullable=False),
        sa.Column("version_tag", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="DRAFT"),
        sa.Column("artifact_path", sa.String(length=500), nullable=False),
        sa.Column("artifact_format", sa.String(length=50), nullable=False, server_default="JOBLIB"),
        sa.Column("artifact_hash_sha256", sa.String(length=64), nullable=False),
        sa.Column("input_schema", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("output_schema", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_by_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id", "version_tag", name="uq_model_versions_model_tag"),
    )
    op.create_index("ix_model_versions_model_id", "model_versions", ["model_id"])
    op.create_index("ix_model_versions_version_tag", "model_versions", ["version_tag"])
    op.create_index("ix_model_versions_status", "model_versions", ["status"])
    op.create_index("ix_model_versions_created_by_id", "model_versions", ["created_by_id"])

    # 4. offline_metrics
    op.create_table(
        "offline_metrics",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_version_id", sa.UUID(), nullable=False),
        sa.Column("dataset_split", sa.String(length=50), nullable=False, server_default="test"),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("custom_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_version_id"),
    )
    op.create_index("ix_offline_metrics_model_version_id", "offline_metrics", ["model_version_id"], unique=True)

    # 5. reference_stats
    op.create_table(
        "reference_stats",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_version_id", sa.UUID(), nullable=False),
        sa.Column("num_samples", sa.Integer(), nullable=False),
        sa.Column("feature_stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("prediction_stats", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_version_id"),
    )
    op.create_index("ix_reference_stats_model_version_id", "reference_stats", ["model_version_id"], unique=True)

    # 6. deployments
    op.create_table(
        "deployments",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_id", sa.UUID(), nullable=False),
        sa.Column("environment", sa.String(length=50), nullable=False, server_default="production"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="ACTIVE"),
        sa.Column("strategy", sa.String(length=50), nullable=False, server_default="DIRECT"),
        sa.Column("primary_version_id", sa.UUID(), nullable=False),
        sa.Column("canary_version_id", sa.UUID(), nullable=True),
        sa.Column("traffic_split", sa.Float(), nullable=False, server_default="100.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["canary_version_id"], ["model_versions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["primary_version_id"], ["model_versions.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_deployments_model_id", "deployments", ["model_id"])
    op.create_index("ix_deployments_environment", "deployments", ["environment"])
    op.create_index("ix_deployments_status", "deployments", ["status"])
    op.create_index("ix_deployments_primary_version_id", "deployments", ["primary_version_id"])
    op.create_index("ix_deployments_canary_version_id", "deployments", ["canary_version_id"])

    # 7. inference_requests
    op.create_table(
        "inference_requests",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("model_version_id", sa.UUID(), nullable=False),
        sa.Column("deployment_id", sa.UUID(), nullable=True),
        sa.Column("input_features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("prediction", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("cached", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["deployment_id"], ["deployments.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_inference_requests_request_id", "inference_requests", ["request_id"])
    op.create_index("ix_inference_requests_model_version_id", "inference_requests", ["model_version_id"])
    op.create_index("ix_inference_requests_deployment_id", "inference_requests", ["deployment_id"])
    op.create_index("ix_inference_requests_created_at", "inference_requests", ["created_at"])

    # 8. ground_truth
    op.create_table(
        "ground_truth",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("inference_request_id", sa.UUID(), nullable=False),
        sa.Column("actual_label", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("matched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["inference_request_id"], ["inference_requests.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("inference_request_id"),
    )
    op.create_index("ix_ground_truth_inference_request_id", "ground_truth", ["inference_request_id"], unique=True)

    # 9. online_metrics
    op.create_table(
        "online_metrics",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_version_id", sa.UUID(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("metrics_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_version_id", "window_start", "window_end", name="uq_online_metrics_window"),
    )
    op.create_index("ix_online_metrics_model_version_id", "online_metrics", ["model_version_id"])
    op.create_index("ix_online_metrics_window_start", "online_metrics", ["window_start"])
    op.create_index("ix_online_metrics_window_end", "online_metrics", ["window_end"])

    # 10. perf_metrics
    op.create_table(
        "perf_metrics",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_version_id", sa.UUID(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_requests", sa.Integer(), nullable=False),
        sa.Column("cache_hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("p50_latency_ms", sa.Float(), nullable=False),
        sa.Column("p95_latency_ms", sa.Float(), nullable=False),
        sa.Column("p99_latency_ms", sa.Float(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_perf_metrics_model_version_id", "perf_metrics", ["model_version_id"])
    op.create_index("ix_perf_metrics_window_start", "perf_metrics", ["window_start"])
    op.create_index("ix_perf_metrics_window_end", "perf_metrics", ["window_end"])

    # 11. drift_events
    op.create_table(
        "drift_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_version_id", sa.UUID(), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("drift_detected", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("metric_used", sa.String(length=50), nullable=False, server_default="PSI"),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("overall_drift_score", sa.Float(), nullable=False),
        sa.Column("feature_drift_details", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drift_events_model_version_id", "drift_events", ["model_version_id"])
    op.create_index("ix_drift_events_drift_detected", "drift_events", ["drift_detected"])
    op.create_index("ix_drift_events_created_at", "drift_events", ["created_at"])

    # 12. audit_logs
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("entity_id", sa.String(length=100), nullable=True),
        sa.Column("payload_before", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("payload_after", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_entity_type", "audit_logs", ["entity_type"])
    op.create_index("ix_audit_logs_entity_id", "audit_logs", ["entity_id"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("drift_events")
    op.drop_table("perf_metrics")
    op.drop_table("online_metrics")
    op.drop_table("ground_truth")
    op.drop_table("inference_requests")
    op.drop_table("deployments")
    op.drop_table("reference_stats")
    op.drop_table("offline_metrics")
    op.drop_table("model_versions")
    op.drop_table("models")
    op.drop_table("users")
