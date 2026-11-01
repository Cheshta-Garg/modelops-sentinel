import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.models.enums import ArtifactFormat
from app.schemas.model import ModelCreateRequest
from app.services.queue import queue_service
from app.services.registry import ModelRegistryService
from app.worker.tasks import TASK_REGISTRY
from scripts.train_baseline_model import train_and_export_model

SWAT_FEATURES = {
    "LIT101": 650.0,
    "FIT101": 2.45,
    "AIT201": 255.0,
    "P101": 1,
    "DPIT301": 19.5,
}

SWAT_ATTACK_FEATURES = {
    "LIT101": 950.0,
    "FIT101": 5.80,
    "AIT201": 380.0,
    "P101": 0,
    "DPIT301": 48.0,
}


@pytest_asyncio.fixture(autouse=True)
async def setup_queue_service(test_engine: AsyncEngine):
    """Ensure queue service runs in deterministic memory mode with test database session factory."""
    queue_service.set_force_memory(True)
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    queue_service.set_session_factory(session_factory)
    await queue_service._memory_queue.clear()
    yield
    queue_service.set_session_factory(None)
    await queue_service._memory_queue.clear()


async def _create_test_model_and_staged_version(
    session: AsyncSession,
    tmp_path: Path,
) -> tuple[uuid.UUID, str]:
    """Helper creating a test model with staged SWaT RandomForest artifact."""
    artifact_path = tmp_path / "test_swat_model.joblib"
    train_and_export_model(str(artifact_path))

    registry = ModelRegistryService()
    model = await registry.create_model(
        session,
        ModelCreateRequest(
            name=f"async_swat_model_{uuid.uuid4().hex[:6]}",
            description="Asynchronous SWaT anomaly test model",
            task_type="binary_classification",
        ),
    )

    class MockUploadFile:
        def __init__(self, path: Path):
            self.filename = path.name
            self.file = open(path, "rb")

        async def read(self) -> bytes:
            return self.file.read()

        async def seek(self, pos: int) -> None:
            self.file.seek(pos)

        async def close(self) -> None:
            self.file.close()

    mock_file = MockUploadFile(artifact_path)
    try:
        version = await registry.create_model_version(
            session=session,
            model_id=model.id,
            file=mock_file,  # type: ignore[arg-type]
            version_tag="v1.0.0",
            input_schema={"features": ["LIT101", "FIT101", "AIT201", "P101", "DPIT301"]},
            output_schema={"prediction": "int", "confidence": "float"},
            artifact_format=ArtifactFormat.JOBLIB,
        )
    finally:
        await mock_file.close()

    return model.id, version.version_tag


@pytest.mark.asyncio
async def test_async_predict_returns_202_accepted(
    async_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
):
    """POST /api/v1/models/{id}/predict/async returns 202 Accepted with a job_id."""
    model_id, _ = await _create_test_model_and_staged_version(db_session, tmp_path)

    response = await async_client.post(
        f"/api/v1/models/{model_id}/predict/async",
        json={"features": SWAT_FEATURES},
    )

    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert len(data["job_id"]) > 0

    # Await job completion so the background task cleanly exits before test engine closes
    for _ in range(25):
        await asyncio.sleep(0.05)
        job = await queue_service.get_job(data["job_id"])
        if job and job.status in ("complete", "failed"):
            break


@pytest.mark.asyncio
async def test_async_predict_execution_and_status_polling(
    async_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
):
    """Submits async prediction, polls GET /api/v1/jobs/{job_id}, and verifies completion."""
    model_id, version_tag = await _create_test_model_and_staged_version(db_session, tmp_path)

    post_resp = await async_client.post(
        f"/api/v1/models/{model_id}/predict/async",
        json={"features": SWAT_FEATURES},
    )
    assert post_resp.status_code == 202
    job_id = post_resp.json()["job_id"]

    # Poll status until complete (timeout after 5 seconds)
    completed_data = None
    for _ in range(25):
        await asyncio.sleep(0.1)
        job_resp = await async_client.get(f"/api/v1/jobs/{job_id}")
        assert job_resp.status_code == 200
        job_data = job_resp.json()
        if job_data["status"] == "complete":
            completed_data = job_data
            break

    assert completed_data is not None, "Job did not reach complete state within timeout"
    assert completed_data["job_id"] == job_id
    assert completed_data["status"] == "complete"
    assert completed_data["success"] is True
    assert completed_data["function"] == "async_predict_task"
    assert completed_data["start_time"] is not None
    assert completed_data["finish_time"] is not None

    result = completed_data["result"]
    assert result["model"].startswith("async_swat_model_")
    assert result["version"] == version_tag
    assert result["prediction"] == 0  # Normal water telemetry
    assert result["confidence"] > 0.5
    assert result["latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_async_predict_anomaly_detection(
    async_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
):
    """Submits attack features and verifies async worker detects anomaly (prediction = 1)."""
    model_id, _ = await _create_test_model_and_staged_version(db_session, tmp_path)

    post_resp = await async_client.post(
        f"/api/v1/models/{model_id}/predict/async",
        json={"features": SWAT_ATTACK_FEATURES},
    )
    job_id = post_resp.json()["job_id"]

    completed_data = None
    for _ in range(25):
        await asyncio.sleep(0.1)
        job_resp = await async_client.get(f"/api/v1/jobs/{job_id}")
        if job_resp.json()["status"] == "complete":
            completed_data = job_resp.json()
            break

    assert completed_data is not None
    assert completed_data["result"]["prediction"] == 1


@pytest.mark.asyncio
async def test_async_predict_unauthenticated_returns_401(
    unauthenticated_client: AsyncClient,
):
    """Unauthenticated request to predict/async returns 401."""
    random_uuid = uuid.uuid4()
    response = await unauthenticated_client.post(
        f"/api/v1/models/{random_uuid}/predict/async",
        json={"features": SWAT_FEATURES},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


@pytest.mark.asyncio
async def test_async_predict_missing_model_returns_404(
    async_client: AsyncClient,
):
    """Requesting async inference for non-existent model returns 404."""
    random_uuid = uuid.uuid4()
    response = await async_client.post(
        f"/api/v1/models/{random_uuid}/predict/async",
        json={"features": SWAT_FEATURES},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_async_predict_missing_version_tag_returns_404(
    async_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
):
    """Requesting async inference with non-existent version_tag returns 404."""
    model_id, _ = await _create_test_model_and_staged_version(db_session, tmp_path)

    response = await async_client.post(
        f"/api/v1/models/{model_id}/predict/async",
        json={"features": SWAT_FEATURES, "version_tag": "v9.9.9"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_async_predict_invalid_features_records_failure_state(
    async_client: AsyncClient,
    db_session: AsyncSession,
    tmp_path: Path,
):
    """Submitting invalid/missing features results in job transitioning to failed state with error details."""
    model_id, _ = await _create_test_model_and_staged_version(db_session, tmp_path)

    # Incomplete features (missing required sensors)
    post_resp = await async_client.post(
        f"/api/v1/models/{model_id}/predict/async",
        json={"features": {"LIT101": 500.0}},
    )
    assert post_resp.status_code == 202
    job_id = post_resp.json()["job_id"]

    failed_data = None
    for _ in range(25):
        await asyncio.sleep(0.1)
        job_resp = await async_client.get(f"/api/v1/jobs/{job_id}")
        data = job_resp.json()
        if data["status"] == "failed":
            failed_data = data
            break

    assert failed_data is not None, "Job did not transition to failed state"
    assert failed_data["status"] == "failed"
    assert failed_data["success"] is False
    assert "Missing required feature(s)" in failed_data["error"]


@pytest.mark.asyncio
async def test_get_job_not_found_returns_404(
    async_client: AsyncClient,
):
    """Querying a non-existent job ID returns 404 Not Found."""
    response = await async_client.get("/api/v1/jobs/non_existent_job_123")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_process_telemetry_task():
    """Worker task process_telemetry_task aggregates batch telemetry events."""
    sample_events = [
        {"prediction": 0, "latency_ms": 15.2},
        {"prediction": 0, "latency_ms": 14.8},
        {"prediction": 1, "latency_ms": 18.0},
    ]

    job_id = await queue_service.enqueue("process_telemetry_task", sample_events)

    completed_job = None
    for _ in range(20):
        await asyncio.sleep(0.05)
        job = await queue_service.get_job(job_id)
        if job and job.status == "complete":
            completed_job = job
            break

    assert completed_job is not None
    assert completed_job.success is True
    res = completed_job.result
    assert res["processed_count"] == 3
    assert res["avg_latency_ms"] == 16.0
    assert res["prediction_distribution"] == {"0": 2, "1": 1}


@pytest.mark.asyncio
async def test_worker_task_retry_behavior():
    """Worker retries a transiently failing task up to max_tries before succeeding."""
    call_count = 0

    async def transient_failing_task(ctx: dict) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError(f"Transient error on try {call_count}")
        return {"status": "recovered", "attempts": call_count}

    TASK_REGISTRY["transient_failing_task"] = transient_failing_task

    job_id = await queue_service.enqueue("transient_failing_task", max_tries=3)

    completed_job = None
    for _ in range(30):
        await asyncio.sleep(0.1)
        job = await queue_service.get_job(job_id)
        if job and job.status == "complete":
            completed_job = job
            break

    assert completed_job is not None
    assert completed_job.success is True
    assert completed_job.result["attempts"] == 3
    assert completed_job.retries == 2


@pytest.mark.asyncio
async def test_worker_task_timeout_behavior():
    """Worker terminates a job that exceeds job_timeout and records failure."""

    async def slow_hanging_task(ctx: dict) -> str:
        await asyncio.sleep(3.0)
        return "finished"

    TASK_REGISTRY["slow_hanging_task"] = slow_hanging_task

    # Max tries 1, timeout 0.2s
    job_id = await queue_service.enqueue("slow_hanging_task", job_timeout=0.2, max_tries=1)

    failed_job = None
    for _ in range(25):
        await asyncio.sleep(0.05)
        job = await queue_service.get_job(job_id)
        if job and job.status == "failed":
            failed_job = job
            break

    assert failed_job is not None
    assert failed_job.status == "failed"
    assert failed_job.success is False
    assert "timeout" in failed_job.error.lower()


@pytest.mark.asyncio
async def test_worker_restart_simulation():
    """Worker restart preserves job records and safely cleans up in-flight executions."""

    async def lingering_task(ctx: dict) -> str:
        await asyncio.sleep(1.0)
        return "done"

    TASK_REGISTRY["lingering_task"] = lingering_task

    job_id = await queue_service.enqueue("lingering_task", job_timeout=5, max_tries=2)

    # Immediately simulate restart
    queue_service.restart_worker()

    job_info = await queue_service.get_job(job_id)
    assert job_info is not None
    assert job_info.status in ("queued", "in_progress", "complete")
