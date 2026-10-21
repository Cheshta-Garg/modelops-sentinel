from typing import Any

from pydantic import BaseModel, Field


class InferenceRequestPayload(BaseModel):
    """Payload submitted for synchronous model prediction."""

    features: dict[str, Any] | list[Any] = Field(
        ...,
        description="Input feature vector as a named key-value dictionary or ordered numerical list.",
        examples=[
            {
                "LIT101": 520.5,
                "FIT101": 2.45,
                "AIT201": 260.1,
                "P101": 1,
                "DPIT301": 19.8,
            }
        ],
    )
    version_tag: str | None = Field(
        default=None,
        description="Optional specific version tag (e.g. 'v1.0.0'). Defaults to latest STAGING or ACTIVE version.",
    )


class InferenceResponse(BaseModel):
    """Synchronous model prediction response contract."""

    request_id: str = Field(description="Unique correlation ID tracing this request.")
    model: str = Field(description="Logical model identifier.")
    version: str = Field(description="Model version tag executing this inference.")
    prediction: Any = Field(description="Model output class label or regression value.")
    confidence: float = Field(
        description="Predicted class probability or confidence score (0.0 to 1.0)."
    )
    latency_ms: float = Field(description="Total end-to-end inference execution latency in milliseconds.")
    cached: bool = Field(default=False, description="Indicates if result was served from Redis cache.")
