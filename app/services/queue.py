import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from arq.connections import ArqRedis, create_pool
from arq.jobs import Job, JobStatus
from arq.worker import Retry

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import track_async_job
from app.schemas.job import JobStatusResponse
from app.worker.settings import WorkerSettings
from app.worker.tasks import TASK_REGISTRY

logger = get_logger(__name__)
settings = get_settings()


class MemoryQueueBackend:
    """In-memory async queue and worker backend.

    Guarantees deterministic, Zero-Docker operation for development, CI, and automated testing.
    Replicates arq semantics: lifecycle states, retries, backoff, timeouts, and failure tracking.
    """

    def __init__(self, session_factory_provider: Any = None) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._session_factory_provider = session_factory_provider

    async def clear(self) -> None:
        """Reset internal job registry and cancel running tasks (useful for test isolation)."""
        tasks = [task for task in self._active_tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_tasks.clear()
        self._jobs.clear()

    async def enqueue(
        self,
        func_name: str,
        *args: Any,
        job_id: str | None = None,
        job_timeout: int = 30,
        max_tries: int = 3,
        **kwargs: Any,
    ) -> str:
        """Enqueue a background job for asynchronous in-memory execution."""
        jid = job_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        self._jobs[jid] = {
            "job_id": jid,
            "function": func_name,
            "args": args,
            "kwargs": kwargs,
            "status": "queued",
            "enqueue_time": now,
            "start_time": None,
            "finish_time": None,
            "success": None,
            "result": None,
            "error": None,
            "job_try": 0,
            "max_tries": max_tries,
            "timeout": job_timeout,
        }

        task = asyncio.create_task(self._run_job(jid))
        self._active_tasks[jid] = task
        return jid

    async def _run_job(self, job_id: str) -> None:
        """Worker execution loop supporting timeouts, retries, and error recording."""
        record = self._jobs.get(job_id)
        if not record:
            return

        func_name = record["function"]
        fn = TASK_REGISTRY.get(func_name)
        if not fn:
            record["status"] = "failed"
            record["success"] = False
            record["error"] = f"Unknown task function: '{func_name}'"
            record["finish_time"] = datetime.now(timezone.utc)
            return

        session_factory = (
            self._session_factory_provider() if self._session_factory_provider else None
        )

        while record["job_try"] < record["max_tries"]:
            record["job_try"] += 1
            record["status"] = "in_progress"
            record["start_time"] = datetime.now(timezone.utc)

            ctx = {
                "job_id": job_id,
                "job_try": record["job_try"],
                "max_tries": record["max_tries"],
                "session_factory": session_factory,
            }

            try:
                # Execute task bounded by timeout
                result = await asyncio.wait_for(
                    fn(ctx, *record["args"], **record["kwargs"]),
                    timeout=float(record["timeout"]),
                )
                record["status"] = "complete"
                record["success"] = True
                record["result"] = result
                record["finish_time"] = datetime.now(timezone.utc)
                track_async_job(func_name, "completed")
                return
            except asyncio.CancelledError:
                record["status"] = "failed"
                record["success"] = False
                record["error"] = "Job cancelled due to worker restart or abort"
                record["finish_time"] = datetime.now(timezone.utc)
                track_async_job(func_name, "failed")
                return
            except asyncio.TimeoutError:
                logger.warning(
                    f"[MemoryWorker] Job '{job_id}' timed out on try {record['job_try']}/{record['max_tries']}"
                )
                if record["job_try"] < record["max_tries"]:
                    record["status"] = "deferred"
                    await asyncio.sleep(0.2)
                    continue
                record["status"] = "failed"
                record["success"] = False
                record["error"] = f"Job exceeded execution timeout of {record['timeout']} seconds"
                record["finish_time"] = datetime.now(timezone.utc)
                track_async_job(func_name, "failed")
                return
            except Retry as r:
                logger.warning(
                    f"[MemoryWorker] Job '{job_id}' requested retry on try {record['job_try']}/{record['max_tries']}"
                )
                if record["job_try"] < record["max_tries"]:
                    record["status"] = "deferred"
                    await asyncio.sleep(r.defer_score or 0.2)
                    continue
                record["status"] = "failed"
                record["success"] = False
                record["error"] = f"Max retries ({record['max_tries']}) exceeded"
                record["finish_time"] = datetime.now(timezone.utc)
                track_async_job(func_name, "failed")
                return
            except Exception as exc:
                logger.warning(
                    f"[MemoryWorker] Job '{job_id}' failed on try {record['job_try']}/{record['max_tries']}: {exc}"
                )
                if record["job_try"] < record["max_tries"]:
                    record["status"] = "deferred"
                    await asyncio.sleep(0.2)
                    continue
                record["status"] = "failed"
                record["success"] = False
                record["error"] = str(exc)
                record["finish_time"] = datetime.now(timezone.utc)
                track_async_job(func_name, "failed")
                return

    async def get_job(self, job_id: str) -> JobStatusResponse | None:
        """Fetch job status and payload from in-memory registry."""
        record = self._jobs.get(job_id)
        if not record:
            return None

        return JobStatusResponse(
            job_id=record["job_id"],
            status=record["status"],
            function=record["function"],
            enqueue_time=record["enqueue_time"],
            start_time=record["start_time"],
            finish_time=record["finish_time"],
            success=record["success"],
            result=record["result"],
            error=record["error"],
            retries=max(0, record["job_try"] - 1),
        )

    def restart_worker(self) -> None:
        """Simulate worker restart by cancelling active runs and re-queuing unfinished jobs."""
        for jid, task in list(self._active_tasks.items()):
            if not task.done():
                task.cancel()
            rec = self._jobs.get(jid)
            if rec and rec["status"] in ("in_progress", "deferred", "queued"):
                rec["status"] = "queued"
                new_task = asyncio.create_task(self._run_job(jid))
                self._active_tasks[jid] = new_task


class QueueService:
    """Unified Queue Service managing job submission and status queries across Redis/arq and Memory."""

    def __init__(self) -> None:
        self._session_factory: Any = None
        self._memory_queue = MemoryQueueBackend(
            session_factory_provider=lambda: self._session_factory
        )
        self._arq_pool: ArqRedis | None = None
        self._force_memory = False
        self._redis_offline_until: float = 0.0

    def set_session_factory(self, factory: Any) -> None:
        """Inject scoped database session factory for background execution."""
        self._session_factory = factory

    def set_force_memory(self, enabled: bool) -> None:
        """Force memory mode (useful for fast, isolated unit tests)."""
        self._force_memory = enabled

    async def _get_arq_pool(self) -> ArqRedis | None:
        """Attempt to obtain an ArqRedis pool with circuit breaker protection."""
        if self._force_memory:
            return None

        now = time.time()
        if now < self._redis_offline_until:
            return None

        if self._arq_pool is not None:
            return self._arq_pool

        try:
            self._arq_pool = await create_pool(
                WorkerSettings.redis_settings,
                default_queue_name="arq:queue",
            )
            logger.info("Successfully connected to Redis arq queue pool.")
            return self._arq_pool
        except Exception as exc:
            self._redis_offline_until = now + 15.0
            logger.debug(
                f"Redis queue unavailable, engaging memory queue fallback (circuit breaker 15s): {exc}"
            )
            return None

    async def enqueue(
        self,
        func_name: str,
        *args: Any,
        job_id: str | None = None,
        job_timeout: int = 60,
        max_tries: int = 3,
        **kwargs: Any,
    ) -> str:
        """Enqueue task into Redis arq queue if online, or fallback to in-memory queue."""
        jid = job_id or uuid.uuid4().hex
        pool = await self._get_arq_pool()

        track_async_job(func_name, "enqueued")

        if pool is not None:
            try:
                await pool.enqueue_job(
                    func_name,
                    *args,
                    _job_id=jid,
                    _timeout=job_timeout,
                    **kwargs,
                )
                logger.info(f"Enqueued job '{jid}' to Redis arq queue ({func_name})")
                return jid
            except Exception as exc:
                logger.warning(
                    f"Failed to enqueue to Redis pool, falling back to memory queue: {exc}"
                )

        return await self._memory_queue.enqueue(
            func_name,
            *args,
            job_id=jid,
            job_timeout=job_timeout,
            max_tries=max_tries,
            **kwargs,
        )

    async def get_job(self, job_id: str) -> JobStatusResponse | None:
        """Query job status from Redis arq or in-memory backend."""
        pool = await self._get_arq_pool()

        if pool is not None:
            try:
                job = Job(job_id=job_id, redis=pool)
                status_raw = await job.status()
                if status_raw == JobStatus.not_found:
                    return await self._memory_queue.get_job(job_id)

                status_str = status_raw.value if hasattr(status_raw, "value") else str(status_raw)
                res_info = None
                if status_raw == JobStatus.complete:
                    res_info = await job.result_info()

                if res_info:
                    return JobStatusResponse(
                        job_id=job_id,
                        status=status_str if res_info.success else "failed",
                        function=res_info.function,
                        enqueue_time=res_info.enqueue_time,
                        start_time=res_info.start_time,
                        finish_time=res_info.finish_time,
                        success=res_info.success,
                        result=res_info.result if res_info.success else None,
                        error=str(res_info.result) if not res_info.success else None,
                        retries=max(0, res_info.job_try - 1),
                    )

                return JobStatusResponse(
                    job_id=job_id,
                    status=status_str,
                    function=None,
                    enqueue_time=None,
                    start_time=None,
                    finish_time=None,
                    success=None,
                    result=None,
                    error=None,
                    retries=0,
                )
            except Exception as exc:
                logger.debug(f"Redis get_job failed for '{job_id}': {exc}")

        return await self._memory_queue.get_job(job_id)

    def restart_worker(self) -> None:
        """Trigger worker restart simulation."""
        self._memory_queue.restart_worker()


# Singleton QueueService instance
queue_service = QueueService()
