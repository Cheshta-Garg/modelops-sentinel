import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import ArtifactFormat, ModelStatus


class ModelCreateRequest(BaseModel):
    """Schema for registering a new logical ML model."""

    name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique identifier name for the model (alphanumeric, hyphens, underscores).",
        examples=["fraud_detector_v1"],
    )
    description: str | None = Field(
        default=None,
        max_length=1000,
        description="Detailed description of model purpose and business problem.",
    )
    task_type: str = Field(
        default="binary_classification",
        max_length=50,
        description="Machine learning problem type (e.g. binary_classification, regression).",
        examples=["binary_classification"],
    )


class ModelVersionSummaryResponse(BaseModel):
    """Lightweight summary of a model version iteration."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version_tag: str
    status: ModelStatus
    artifact_format: ArtifactFormat
    created_at: datetime


class ModelResponse(BaseModel):
    """Detailed metadata for a registered model."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    task_type: str
    created_at: datetime
    updated_at: datetime
    versions_count: int = 0


class ModelDetailResponse(ModelResponse):
    """Complete model entity including version iterations."""

    versions: list[ModelVersionSummaryResponse] = Field(default_factory=list)


class ModelVersionResponse(BaseModel):
    """Complete metadata and schema specifications for a validated model version."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    model_id: uuid.UUID
    version_tag: str
    status: ModelStatus
    artifact_path: str
    artifact_format: ArtifactFormat
    artifact_hash_sha256: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    created_at: datetime
    updated_at: datetime
