from datetime import datetime, timezone

from arq.connections import RedisSettings

from app.core.config import get_settings
from app.core.logging import get_logger
from app.worker.tasks import async_predict_task, process_telemetry_task

logger = get_logger(__name__)
settings = get_settings()


async def startup(ctx: dict) -> None:
    """Worker initialization lifecycle hook."""
    logger.info("Initializing ModelOps Sentinel background worker...")
    ctx["worker_id"] = "modelops-sentinel-worker-1"
    ctx["started_at"] = datetime.now(timezone.utc).isoformat()


async def shutdown(ctx: dict) -> None:
    """Worker shutdown lifecycle hook."""
    logger.info("Shutting down ModelOps Sentinel background worker gracefully...")


class WorkerSettings:
    """Arq Worker Configuration.

    Run via terminal:
        arq app.worker.settings.WorkerSettings
    """

    functions = [async_predict_task, process_telemetry_task]
    redis_settings = RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        database=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
        conn_timeout=1,
        conn_retries=1,
    )
    job_timeout = 60  # Maximum job execution time (seconds) before timeout
    max_tries = 3  # Maximum execution attempts / retries on failure
    retry_delay = 2  # Backoff delay (seconds) before retrying a failed task
    on_startup = startup
    on_shutdown = shutdown
