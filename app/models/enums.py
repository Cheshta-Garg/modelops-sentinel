import enum


class UserRole(str, enum.Enum):
    """RBAC User Roles."""

    ADMIN = "admin"
    ENGINEER = "engineer"
    VIEWER = "viewer"
    OPERATOR = "operator"

    @classmethod
    def _missing_(cls, value: object) -> enum.Enum | None:
        if isinstance(value, str):
            val_lower = value.lower()
            for member in cls:
                if member.value == val_lower or member.name.lower() == val_lower:
                    return member
        return None



class ModelStatus(str, enum.Enum):
    """Lifecycle status for a model version state machine."""

    REGISTERED = "REGISTERED"
    VALIDATING = "VALIDATING"
    STAGING = "STAGING"
    FAILED = "FAILED"
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
