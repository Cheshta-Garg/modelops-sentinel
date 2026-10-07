import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import GUID, Base, JSONBCompat, utc_now

if TYPE_CHECKING:
    from app.models.deployment import Deployment
    from app.models.model import ModelVersion


class InferenceRequest(Base):
    """Immutable audit record of a served prediction query."""

    __tablename__ = "inference_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    request_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    deployment_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID,
        ForeignKey("deployments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    input_features: Mapped[dict[str, Any]] = mapped_column(
        JSONBCompat,
        nullable=False,
    )
    prediction: Mapped[dict[str, Any]] = mapped_column(
        JSONBCompat,
        nullable=False,
    )
    latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    cached: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion",
        back_populates="inference_requests",
    )
    deployment: Mapped["Deployment | None"] = relationship(
        "Deployment",
        back_populates="inference_requests",
    )
    ground_truth: Mapped["GroundTruth | None"] = relationship(
        "GroundTruth",
        back_populates="inference_request",
        uselist=False,
        lazy="selectin",
        cascade="all, delete-orphan",
    )


class GroundTruth(Base):
    """Real-world outcome label joined back to an earlier prediction for accuracy auditing."""

    __tablename__ = "ground_truth"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    inference_request_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("inference_requests.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    actual_label: Mapped[dict[str, Any]] = mapped_column(
        JSONBCompat,
        nullable=False,
    )
    matched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    # Relationships
    inference_request: Mapped["InferenceRequest"] = relationship(
        "InferenceRequest",
        back_populates="ground_truth",
    )
