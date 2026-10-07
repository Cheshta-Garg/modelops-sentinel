import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Enum, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import GUID, Base, JSONBCompat, TimestampMixin
from app.models.enums import ArtifactFormat, ModelStatus

if TYPE_CHECKING:
    from app.models.deployment import Deployment
    from app.models.inference import InferenceRequest
    from app.models.metrics import (
        OfflineMetrics,
        OnlineMetrics,
        PerfMetrics,
        ReferenceStats,
    )
    from app.models.monitoring import DriftEvent
    from app.models.user import User


class Model(Base, TimestampMixin):
    """Logical Machine Learning Model entity (e.g. churn_detector)."""

    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    task_type: Mapped[str] = mapped_column(
        String(50),
        default="binary_classification",
        nullable=False,
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Relationships
    created_by: Mapped["User | None"] = relationship(
        "User",
        back_populates="models",
    )
    versions: Mapped[list["ModelVersion"]] = relationship(
        "ModelVersion",
        back_populates="model",
        cascade="all, delete-orphan",
    )
    deployments: Mapped[list["Deployment"]] = relationship(
        "Deployment",
        back_populates="model",
        cascade="all, delete-orphan",
    )


class ModelVersion(Base, TimestampMixin):
    """Immutable, versioned ML model artifact iteration (e.g. v1.0.0)."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "version_tag", name="uq_model_versions_model_tag"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_tag: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus, native_enum=False),
        default=ModelStatus.DRAFT,
        nullable=False,
        index=True,
    )
    artifact_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    artifact_format: Mapped[ArtifactFormat] = mapped_column(
        Enum(ArtifactFormat, native_enum=False),
        default=ArtifactFormat.JOBLIB,
        nullable=False,
    )
    artifact_hash_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    input_schema: Mapped[dict[str, Any]] = mapped_column(
        JSONBCompat,
        nullable=False,
    )
    output_schema: Mapped[dict[str, Any]] = mapped_column(
        JSONBCompat,
        nullable=False,
    )
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Relationships
    model: Mapped["Model"] = relationship(
        "Model",
        back_populates="versions",
    )
    created_by: Mapped["User | None"] = relationship(
        "User",
        back_populates="model_versions",
    )
    offline_metrics: Mapped["OfflineMetrics | None"] = relationship(
        "OfflineMetrics",
        back_populates="model_version",
        uselist=False,
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    reference_stats: Mapped["ReferenceStats | None"] = relationship(
        "ReferenceStats",
        back_populates="model_version",
        uselist=False,
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    inference_requests: Mapped[list["InferenceRequest"]] = relationship(
        "InferenceRequest",
        back_populates="model_version",
        cascade="all, delete-orphan",
    )
    online_metrics: Mapped[list["OnlineMetrics"]] = relationship(
        "OnlineMetrics",
        back_populates="model_version",
        cascade="all, delete-orphan",
    )
    perf_metrics: Mapped[list["PerfMetrics"]] = relationship(
        "PerfMetrics",
        back_populates="model_version",
        cascade="all, delete-orphan",
    )
    drift_events: Mapped[list["DriftEvent"]] = relationship(
        "DriftEvent",
        back_populates="model_version",
        cascade="all, delete-orphan",
    )
