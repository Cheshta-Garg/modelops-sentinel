"""Pydantic schemas for statistical drift monitoring and reports."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DriftEventItem(BaseModel):
    """Schema for an individual statistical drift evaluation event."""

    id: uuid.UUID
    feature: str = Field(..., description="Feature name evaluated")
    method: str = Field(..., description="Statistical test method (PSI or KS_TEST)")
    score: float = Field(..., description="Calculated statistical drift score")
    threshold: float = Field(..., description="Threshold applied to determine drift")
    severity: str = Field(..., description="Drift severity level (NONE, MEDIUM, HIGH)")
    drift_detected: bool = Field(..., description="Whether the feature breached drift threshold")
    detected_at: datetime = Field(..., description="Timestamp of evaluation")

    model_config = ConfigDict(from_attributes=True)


class DriftSummary(BaseModel):
    """Aggregated drift summary across recent evaluations."""

    total_evaluations: int = Field(default=0, description="Total individual feature evaluations")
    drifted_features_count: int = Field(
        default=0, description="Count of features flagged as drifted"
    )
    highest_severity: str = Field(
        default="NONE", description="Highest severity observed (NONE, MEDIUM, HIGH)"
    )


class ModelDriftResponse(BaseModel):
    """API response contract for GET /api/v1/models/{id}/drift."""

    model_id: uuid.UUID
    model_name: str
    version_tag: str | None = None
    reference_stats_configured: bool = Field(
        ..., description="Whether baseline reference statistics exist"
    )
    drift_detected: bool = Field(..., description="Whether any feature has drifted")
    summary: DriftSummary
    events: list[DriftEventItem] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
