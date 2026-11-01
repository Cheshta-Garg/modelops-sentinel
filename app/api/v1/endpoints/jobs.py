from fastapi import APIRouter, Depends, status

from app.core.auth import get_current_user
from app.core.errors import NotFoundError
from app.models.user import User
from app.schemas.job import JobStatusResponse
from app.services.queue import queue_service

router = APIRouter(prefix="/jobs", tags=["Asynchronous Jobs & Tasks"])


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get background job status and result",
    description="Fetches current lifecycle status, retry counts, execution result, or error trace of an enqueued job.",
)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> JobStatusResponse:
    """Query current execution status or result of a background job."""
    job_info = await queue_service.get_job(job_id)
    if not job_info:
        raise NotFoundError(message=f"Job with ID '{job_id}' was not found.")

    return job_info
