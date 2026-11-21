"""Pydantic schemas for delayed ground-truth feedback and performance evaluation."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeedbackCreateRequest(BaseModel):
    """Input payload for POST /api/v1/models/{id}/feedback."""

    request_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Correlation request_id returned by a previous inference prediction.",
        examples=["req-trace-abc-123"],
    )
    label: int | float | str | bool = Field(
        ...,
        description="Observed real-world ground-truth outcome label (e.g. 0, 1, 'fraud').",
        examples=[1],
    )

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("request_id must not be empty or whitespace.")
        return stripped


class FeedbackResponse(BaseModel):
    """Response contract returned after recording delayed ground-truth feedback."""

    id: uuid.UUID
    model_id: uuid.UUID
    model_version_id: uuid.UUID
    version_tag: str
    inference_request_id: uuid.UUID
    request_id: str
    predicted_label: Any
    actual_label: Any
    is_correct: bool
    feedback_delay_seconds: float = Field(
        ...,
        description="Elapsed seconds between original inference and ground-truth arrival.",
    )
    matched_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConfusionMatrixSchema(BaseModel):
    """Structured confusion matrix for binary or multiclass classification."""

    labels: list[Any] = Field(..., description="Ordered class labels corresponding to matrix axes.")
    matrix: list[list[int]] = Field(
        ...,
        description="2D confusion matrix where rows are actual labels and columns are predicted labels.",
    )
    tp: int | None = Field(default=None, description="True positives (binary classification).")
    tn: int | None = Field(default=None, description="True negatives (binary classification).")
    fp: int | None = Field(default=None, description="False positives (binary classification).")
    fn: int | None = Field(default=None, description="False negatives (binary classification).")
    averaging: str | None = Field(
        default=None,
        description="Averaging strategy used for precision/recall/F1 (binary or weighted).",
    )


class PerfMetricItem(BaseModel):
    """Individual delayed ground-truth performance evaluation snapshot from perf_metrics."""

    id: uuid.UUID
    model_version_id: uuid.UUID
    window_start: datetime
    window_end: datetime
    sample_count: int = Field(
        ...,
        description="Number of matched (prediction, ground_truth) pairs evaluated.",
    )
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: ConfusionMatrixSchema
    evaluated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ModelPerformanceResponse(BaseModel):
    """Response contract for GET /api/v1/models/{id}/performance (delayed ground-truth only)."""

    metric_category: str = Field(
        default="delayed_ground_truth_performance",
        description="Explicit metric category isolating delayed ground-truth from offline and online metrics.",
    )
    model_id: uuid.UUID
    model_name: str
    version_tag: str | None = None
    total_inference_requests: int = Field(
        default=0,
        description="Total logged inference requests for this model version.",
    )
    matched_ground_truth_count: int = Field(
        default=0,
        description="Total inference requests that have received delayed ground-truth labels.",
    )
    label_coverage_ratio: float = Field(
        default=0.0,
        description="Fraction of served inferences with confirmed ground-truth feedback (0.0 to 1.0).",
    )
    latest_metrics: PerfMetricItem | None = None
    history: list[PerfMetricItem] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
