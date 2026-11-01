from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AsyncJobResponse(BaseModel):
    """Response returned when an asynchronous inference or background task is accepted."""

    job_id: str = Field(
        ...,
        description="Unique identifier for the enqueued background job.",
        examples=["b9f71c3e-8f24-4f51-bca2-892c57f201e8"],
    )


class JobStatusResponse(BaseModel):
    """Detailed status, timing, and result payload of an asynchronous background job."""

    job_id: str = Field(
        ...,
        description="Unique identifier of the queried job.",
        examples=["b9f71c3e-8f24-4f51-bca2-892c57f201e8"],
    )
    status: str = Field(
        ...,
        description="Current lifecycle status: queued, in_progress, complete, failed, deferred, not_found.",
        examples=["complete"],
    )
    function: str | None = Field(
        default=None,
        description="Name of the worker task function executing this job.",
        examples=["async_predict_task"],
    )
    enqueue_time: datetime | None = Field(
        default=None,
        description="UTC timestamp when the job was accepted into the queue.",
    )
    start_time: datetime | None = Field(
        default=None,
        description="UTC timestamp when worker picked up and started executing the job.",
    )
    finish_time: datetime | None = Field(
        default=None,
        description="UTC timestamp when the job finished execution.",
    )
    success: bool | None = Field(
        default=None,
        description="Boolean indicating whether execution completed successfully.",
    )
    result: Any | None = Field(
        default=None,
        description="Returned payload upon successful task completion.",
    )
    error: str | None = Field(
        default=None,
        description="Error message, exception details, or failure cause if task failed.",
    )
    retries: int = Field(
        default=0,
        description="Number of execution attempts or retries performed.",
    )
