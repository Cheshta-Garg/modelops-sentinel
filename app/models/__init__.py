"""SQLAlchemy models package re-exporting all domain entities for Base.metadata."""

from app.db.base import Base
from app.models.deployment import Deployment
from app.models.enums import (
    ArtifactFormat,
    DeploymentStatus,
    DeploymentStrategy,
    DriftMetric,
    ModelStatus,
    UserRole,
)
from app.models.inference import GroundTruth, InferenceRequest
from app.models.metrics import (
    OfflineMetrics,
    OnlineMetrics,
    PerfMetrics,
    ReferenceStats,
)
from app.models.model import Model, ModelVersion
from app.models.monitoring import AuditLog, DriftEvent
from app.models.user import User

__all__ = [
    "Base",
    "User",
    "UserRole",
    "Model",
    "ModelStatus",
    "ArtifactFormat",
    "ModelVersion",
    "OfflineMetrics",
    "ReferenceStats",
    "Deployment",
    "DeploymentStatus",
    "DeploymentStrategy",
    "InferenceRequest",
    "GroundTruth",
    "OnlineMetrics",
    "PerfMetrics",
    "DriftEvent",
    "DriftMetric",
    "AuditLog",
]
