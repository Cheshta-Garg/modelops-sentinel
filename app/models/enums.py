import enum


class UserRole(str, enum.Enum):
    """RBAC User Roles."""

    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    VIEWER = "VIEWER"


class ModelStatus(str, enum.Enum):
    """Lifecycle status for a model version."""

    DRAFT = "DRAFT"
    STAGING = "STAGING"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


class ArtifactFormat(str, enum.Enum):
    """Serialized model format."""

    JOBLIB = "JOBLIB"
    ONNX = "ONNX"


class DeploymentStrategy(str, enum.Enum):
    """Deployment traffic routing strategy."""

    DIRECT = "DIRECT"
    CANARY = "CANARY"
    SHADOW = "SHADOW"


class DeploymentStatus(str, enum.Enum):
    """Deployment operational status."""

    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    FAILED_HEALTH_CHECK = "FAILED_HEALTH_CHECK"


class DriftMetric(str, enum.Enum):
    """Statistical metric used to evaluate drift."""

    PSI = "PSI"
    KS_TEST = "KS_TEST"
    HYBRID = "HYBRID"
