import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import GUID, Base, JSONBCompat, utc_now

if TYPE_CHECKING:
    from app.models.model import ModelVersion


class OfflineMetrics(Base):
    """Validation test set evaluation metrics computed prior to promotion."""

    __tablename__ = "offline_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    dataset_split: Mapped[str] = mapped_column(
        String(50),
        default="test",
        nullable=False,
    )
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[float | None] = mapped_column(Float, nullable=True)
    custom_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONBCompat,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion",
        back_populates="offline_metrics",
    )


class ReferenceStats(Base):
    """Baseline feature and target statistical distributions for drift computation."""

    __tablename__ = "reference_stats"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    num_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_stats: Mapped[dict[str, Any]] = mapped_column(
        JSONBCompat,
        nullable=False,
    )
    prediction_stats: Mapped[dict[str, Any] | None] = mapped_column(
        JSONBCompat,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion",
        back_populates="reference_stats",
    )


class OnlineMetrics(Base):
    """Aggregated model prediction accuracy derived from ground-truth matches."""

    __tablename__ = "online_metrics"
    __table_args__ = (
        UniqueConstraint(
            "model_version_id",
            "window_start",
            "window_end",
            name="uq_online_metrics_window",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSONBCompat,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion",
        back_populates="online_metrics",
    )


class PerfMetrics(Base):
    """Operational latency, throughput, and error metrics across sliding windows."""

    __tablename__ = "perf_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    total_requests: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    p50_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    p95_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    p99_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    error_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion",
        back_populates="perf_metrics",
    )
