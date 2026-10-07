import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Enum, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import GUID, Base, TimestampMixin
from app.models.enums import DeploymentStatus, DeploymentStrategy

if TYPE_CHECKING:
    from app.models.inference import InferenceRequest
    from app.models.model import Model, ModelVersion


class Deployment(Base, TimestampMixin):
    """Traffic routing configuration binding active model versions to live endpoints."""

    __tablename__ = "deployments"

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
    environment: Mapped[str] = mapped_column(
        String(50),
        default="production",
        nullable=False,
        index=True,
    )
    status: Mapped[DeploymentStatus] = mapped_column(
        Enum(DeploymentStatus, native_enum=False),
        default=DeploymentStatus.ACTIVE,
        nullable=False,
        index=True,
    )
    strategy: Mapped[DeploymentStrategy] = mapped_column(
        Enum(DeploymentStrategy, native_enum=False),
        default=DeploymentStrategy.DIRECT,
        nullable=False,
    )
    primary_version_id: Mapped[uuid.UUID] = mapped_column(
        GUID,
        ForeignKey("model_versions.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    canary_version_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID,
        ForeignKey("model_versions.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    traffic_split: Mapped[float] = mapped_column(
        Float,
        default=100.0,
        nullable=False,
    )

    # Relationships
    model: Mapped["Model"] = relationship(
        "Model",
        back_populates="deployments",
    )
    primary_version: Mapped["ModelVersion"] = relationship(
        "ModelVersion",
        foreign_keys=[primary_version_id],
        lazy="selectin",
    )
    canary_version: Mapped["ModelVersion | None"] = relationship(
        "ModelVersion",
        foreign_keys=[canary_version_id],
        lazy="selectin",
    )
    inference_requests: Mapped[list["InferenceRequest"]] = relationship(
        "InferenceRequest",
        back_populates="deployment",
    )
