"""Automated test suite for Stage 9: Delayed Ground-Truth Evaluation."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.enums import ArtifactFormat, ModelStatus, UserRole
from app.models.inference import GroundTruth, InferenceRequest
from app.models.metrics import OfflineMetrics, OnlineMetrics, PerfMetrics
from app.models.model import Model, ModelVersion
from app.models.user import User
from app.services.ground_truth import compute_classification_metrics
from app.services.queue import queue_service
from app.worker.tasks import evaluate_ground_truth_task

# ------------------------------------------------------------------------------
# Helper Factory
# ------------------------------------------------------------------------------


async def _create_model_and_version(
    db_session: AsyncSession,
    name: str,
    version_tag: str = "v1.0.0",
) -> tuple[Model, ModelVersion]:
    user = User(
        id=uuid.uuid4(),
        email=f"{name}_{uuid.uuid4().hex[:6]}@modelops.test",
        full_name="GT Tester",
        hashed_password="mock_hashed_password",
        role=UserRole.ENGINEER,
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    model = Model(
        id=uuid.uuid4(),
        name=name,
        description="Test model for delayed ground-truth evaluation",
        task_type="binary_classification",
        created_by_id=user.id,
    )
    db_session.add(model)
    await db_session.flush()

    version = ModelVersion(
        id=uuid.uuid4(),
        model_id=model.id,
        version_tag=version_tag,
        status=ModelStatus.ACTIVE,
        artifact_path=f"/mock/{name}/{version_tag}/model.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="a" * 64,
        input_schema={"features": ["x1", "x2"]},
        output_schema={"prediction": "int"},
        created_by_id=user.id,
    )
    db_session.add(version)
    await db_session.commit()
    return model, version


# ------------------------------------------------------------------------------
# 1. Pure Metric Computation Unit Tests
# ------------------------------------------------------------------------------


def test_compute_binary_classification_metrics_exact() -> None:
    """Verify exact accuracy, precision, recall, F1, and binary confusion matrix."""
    # TP=3, TN=3, FP=1, FN=1 => total=8
    # accuracy = 6/8 = 0.75, precision = 3/4 = 0.75, recall = 3/4 = 0.75, f1 = 0.75
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0]

    result = compute_classification_metrics(y_true, y_pred)

    assert result["sample_count"] == 8
    assert result["accuracy"] == 0.75
    assert result["precision"] == 0.75
    assert result["recall"] == 0.75
    assert result["f1_score"] == 0.75

    cm = result["confusion_matrix"]
    assert cm["labels"] == [0, 1]
    assert cm["matrix"] == [[3, 1], [1, 3]]
    assert cm["tn"] == 3
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 3
    assert cm["averaging"] == "binary"


def test_compute_binary_metrics_zero_division_safe() -> None:
    """Verify zero predicted positives does not raise ZeroDivisionError or produce NaN."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]

    result = compute_classification_metrics(y_true, y_pred)

    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0
    assert result["confusion_matrix"]["tp"] == 0
    assert result["confusion_matrix"]["fn"] == 2
    assert result["confusion_matrix"]["tn"] == 2
    assert result["confusion_matrix"]["fp"] == 0


def test_compute_multiclass_classification_metrics() -> None:
    """Verify 3-class weighted precision/recall/F1 and 3x3 confusion matrix."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]

    result = compute_classification_metrics(y_true, y_pred)

    assert result["sample_count"] == 6
    assert result["accuracy"] == round(5 / 6, 4)
    assert result["confusion_matrix"]["labels"] == [0, 1, 2]
    assert result["confusion_matrix"]["matrix"] == [
        [2, 0, 0],
        [0, 1, 1],
        [0, 0, 2],
    ]
    assert result["confusion_matrix"]["averaging"] == "weighted"


def test_compute_metrics_string_and_bool_label_alignment() -> None:
    """Verify string binary labels and bool/int alignment work cleanly."""
    res_str = compute_classification_metrics(
        y_true=["normal", "anomaly", "anomaly", "normal"],
        y_pred=["normal", "anomaly", "normal", "normal"],
        positive_label="anomaly",
    )
    assert res_str["accuracy"] == 0.75
    assert res_str["precision"] == 1.0
    assert res_str["recall"] == 0.5
    assert res_str["confusion_matrix"]["tp"] == 1
    assert res_str["confusion_matrix"]["fn"] == 1
    assert res_str["confusion_matrix"]["tn"] == 2
    assert res_str["confusion_matrix"]["fp"] == 0

    # Bool labels compared against int predictions (e.g. True == 1, False == 0)
    res_bool = compute_classification_metrics(
        y_true=[True, False, True, False],
        y_pred=[1, 0, 1, 0],
    )
    assert res_bool["accuracy"] == 1.0
    assert res_bool["f1_score"] == 1.0


# ------------------------------------------------------------------------------
# 2. Feedback Ingestion Endpoint Tests: POST /api/v1/models/{id}/feedback
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_delayed_feedback_stores_ground_truth(
    async_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Verify POST /api/v1/models/{id}/feedback stores delayed label in ground_truth."""
    model, version = await _create_model_and_version(db_session, "fraud-detector-gt")

    # Simulate an inference made 2 hours ago
    past_time = datetime.now(timezone.utc) - timedelta(hours=2)
    inf_req = InferenceRequest(
        id=uuid.uuid4(),
        request_id="req-delayed-001",
        model_version_id=version.id,
        input_features={"x1": 10.0, "x2": 20.0},
        prediction={"value": 1, "confidence": 0.91},
        latency_ms=2.4,
        cached=False,
        created_at=past_time,
    )
    db_session.add(inf_req)
    await db_session.commit()

    response = await async_client.post(
        f"/api/v1/models/{model.id}/feedback",
        json={"request_id": "req-delayed-001", "label": 1},
    )
    assert response.status_code == 201
    body = response.json()

    assert body["model_id"] == str(model.id)
    assert body["model_version_id"] == str(version.id)
    assert body["version_tag"] == "v1.0.0"
    assert body["request_id"] == "req-delayed-001"
    assert body["predicted_label"] == 1
    assert body["actual_label"] == 1
    assert body["is_correct"] is True
    assert body["feedback_delay_seconds"] >= 7100.0  # ~2 hours delay

    # Confirm persisted in ground_truth table
    stmt = select(GroundTruth).where(GroundTruth.inference_request_id == inf_req.id)
    gt_row = (await db_session.execute(stmt)).scalar_one_or_none()
    assert gt_row is not None
    assert gt_row.actual_label == {"value": 1}


@pytest.mark.asyncio
async def test_submit_duplicate_feedback_returns_409(
    async_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Verify duplicate feedback for the same request_id is rejected with 409 Conflict."""
    model, version = await _create_model_and_version(db_session, "dup-feedback-model")

    inf_req = InferenceRequest(
        id=uuid.uuid4(),
        request_id="req-dup-001",
        model_version_id=version.id,
        input_features={"x1": 1.0, "x2": 2.0},
        prediction={"value": 0, "confidence": 0.85},
        latency_ms=1.5,
        cached=False,
    )
    db_session.add(inf_req)
    await db_session.commit()

    resp1 = await async_client.post(
        f"/api/v1/models/{model.id}/feedback",
        json={"request_id": "req-dup-001", "label": 0},
    )
    assert resp1.status_code == 201

    resp2 = await async_client.post(
        f"/api/v1/models/{model.id}/feedback",
        json={"request_id": "req-dup-001", "label": 1},
    )
    assert resp2.status_code == 409
    assert resp2.json()["error"]["code"] == "DUPLICATE_FEEDBACK"


@pytest.mark.asyncio
async def test_submit_feedback_not_found_and_cross_model_isolation(
    async_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Verify unknown model, unknown request_id, and cross-model request_id return 404."""
    model_a, version_a = await _create_model_and_version(db_session, "model-a-iso")
    model_b, _ = await _create_model_and_version(db_session, "model-b-iso")

    inf_a = InferenceRequest(
        id=uuid.uuid4(),
        request_id="req-belongs-to-a",
        model_version_id=version_a.id,
        input_features={"x1": 1.0},
        prediction={"value": 1, "confidence": 0.8},
        latency_ms=1.2,
        cached=False,
    )
    db_session.add(inf_a)
    await db_session.commit()

    # Unknown model ID
    r_model = await async_client.post(
        f"/api/v1/models/{uuid.uuid4()}/feedback",
        json={"request_id": "req-belongs-to-a", "label": 1},
    )
    assert r_model.status_code == 404

    # Unknown request_id for Model A
    r_req = await async_client.post(
        f"/api/v1/models/{model_a.id}/feedback",
        json={"request_id": "req-does-not-exist", "label": 1},
    )
    assert r_req.status_code == 404

    # Cross-model attempt: request belongs to Model A, submitted against Model B
    r_cross = await async_client.post(
        f"/api/v1/models/{model_b.id}/feedback",
        json={"request_id": "req-belongs-to-a", "label": 1},
    )
    assert r_cross.status_code == 404


# ------------------------------------------------------------------------------
# 3. Worker Evaluation & Delayed Performance Storage (perf_metrics)
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_evaluate_ground_truth_and_get_performance_api(
    async_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """End-to-end delayed feedback -> worker calculation -> perf_metrics -> GET /performance."""
    model, version = await _create_model_and_version(db_session, "credit-default-gt")

    base_time = datetime.now(timezone.utc) - timedelta(days=3)

    # 10 served predictions; only 8 will receive delayed ground-truth feedback (80% coverage)
    # Matched pairs:
    #   TP=4 (pred=1, actual=1), FP=1 (pred=1, actual=0),
    #   FN=1 (pred=0, actual=1), TN=2 (pred=0, actual=0)
    #   accuracy = 6/8 = 0.75, precision = 4/5 = 0.8, recall = 4/5 = 0.8, f1 = 0.8
    samples = [
        ("req-gt-01", 1, 1),  # TP
        ("req-gt-02", 1, 1),  # TP
        ("req-gt-03", 1, 1),  # TP
        ("req-gt-04", 1, 1),  # TP
        ("req-gt-05", 1, 0),  # FP
        ("req-gt-06", 0, 1),  # FN
        ("req-gt-07", 0, 0),  # TN
        ("req-gt-08", 0, 0),  # TN
        ("req-gt-09", 0, None),  # Unlabeled (delayed outcome not yet known)
        ("req-gt-10", 1, None),  # Unlabeled (delayed outcome not yet known)
    ]

    for idx, (req_id, pred_val, _) in enumerate(samples):
        db_session.add(
            InferenceRequest(
                id=uuid.uuid4(),
                request_id=req_id,
                model_version_id=version.id,
                input_features={"x1": float(idx), "x2": float(idx * 2)},
                prediction={"value": pred_val, "confidence": 0.88},
                latency_ms=2.0,
                cached=False,
                created_at=base_time + timedelta(minutes=idx * 10),
            )
        )
    await db_session.commit()

    # Submit delayed feedback for the 8 resolved outcomes via API
    for req_id, _, actual_val in samples:
        if actual_val is not None:
            resp = await async_client.post(
                f"/api/v1/models/{model.id}/feedback",
                json={"request_id": req_id, "label": actual_val},
            )
            assert resp.status_code == 201

    # Run background worker evaluation task
    test_session_factory = async_sessionmaker(
        bind=db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    worker_result = await evaluate_ground_truth_task(
        ctx={"session_factory": test_session_factory},
        model_version_id=str(version.id),
        window_size=100,
    )

    assert worker_result["status"] == "COMPLETED"
    assert worker_result["metric_category"] == "delayed_ground_truth_performance"
    assert worker_result["sample_count"] == 8
    assert worker_result["accuracy"] == 0.75
    assert worker_result["precision"] == 0.8
    assert worker_result["recall"] == 0.8
    assert worker_result["f1_score"] == 0.8
    assert worker_result["confusion_matrix"]["tp"] == 4
    assert worker_result["confusion_matrix"]["fp"] == 1
    assert worker_result["confusion_matrix"]["fn"] == 1
    assert worker_result["confusion_matrix"]["tn"] == 2
    assert worker_result["confusion_matrix"]["matrix"] == [[2, 1], [1, 4]]

    # Verify persisted in perf_metrics table
    stmt_perf = select(PerfMetrics).where(PerfMetrics.model_version_id == version.id)
    perf_rows = (await db_session.execute(stmt_perf)).scalars().all()
    assert len(perf_rows) == 1
    assert perf_rows[0].accuracy == 0.75
    assert perf_rows[0].f1_score == 0.8

    # Query GET /api/v1/models/{id}/performance
    perf_resp = await async_client.get(f"/api/v1/models/{model.id}/performance")
    assert perf_resp.status_code == 200
    perf_body = perf_resp.json()

    assert perf_body["metric_category"] == "delayed_ground_truth_performance"
    assert perf_body["total_inference_requests"] == 10
    assert perf_body["matched_ground_truth_count"] == 8
    assert perf_body["label_coverage_ratio"] == 0.8
    assert perf_body["latest_metrics"]["accuracy"] == 0.75
    assert perf_body["latest_metrics"]["precision"] == 0.8
    assert perf_body["latest_metrics"]["recall"] == 0.8
    assert perf_body["latest_metrics"]["f1_score"] == 0.8
    assert perf_body["latest_metrics"]["confusion_matrix"]["tp"] == 4


@pytest.mark.asyncio
async def test_worker_no_ground_truth_skips_perf_metrics_write(
    db_session: AsyncSession,
) -> None:
    """Verify worker gracefully skips writing perf_metrics when no ground truth exists."""
    _, version = await _create_model_and_version(db_session, "no-gt-model")

    db_session.add(
        InferenceRequest(
            id=uuid.uuid4(),
            request_id="req-unlabeled-only",
            model_version_id=version.id,
            input_features={"x1": 5.0},
            prediction={"value": 0, "confidence": 0.9},
            latency_ms=1.1,
            cached=False,
        )
    )
    await db_session.commit()

    test_session_factory = async_sessionmaker(
        bind=db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    result = await evaluate_ground_truth_task(
        ctx={"session_factory": test_session_factory},
        model_version_id=str(version.id),
    )

    assert result["status"] == "NO_GROUND_TRUTH"
    assert result["sample_count"] == 0

    perf_rows = (
        (await db_session.execute(select(PerfMetrics).where(PerfMetrics.model_version_id == version.id)))
        .scalars()
        .all()
    )
    assert len(perf_rows) == 0


@pytest.mark.asyncio
async def test_trigger_performance_evaluate_sync_and_async_modes(
    async_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Verify POST /api/v1/models/{id}/performance/evaluate supports both sync and async worker modes."""
    model, version = await _create_model_and_version(db_session, "eval-trigger-model")

    for i, (pred, actual) in enumerate([(1, 1), (0, 0), (1, 0), (0, 0)]):
        inf = InferenceRequest(
            id=uuid.uuid4(),
            request_id=f"req-trig-{i}",
            model_version_id=version.id,
            input_features={"x1": float(i)},
            prediction={"value": pred, "confidence": 0.8},
            latency_ms=1.0,
            cached=False,
        )
        db_session.add(inf)
        await db_session.flush()
        db_session.add(
            GroundTruth(
                inference_request_id=inf.id,
                actual_label={"value": actual},
            )
        )
    await db_session.commit()

    # 1. Sync evaluation via API
    sync_resp = await async_client.post(f"/api/v1/models/{model.id}/performance/evaluate")
    assert sync_resp.status_code == 200
    sync_body = sync_resp.json()
    assert sync_body["status"] == "COMPLETED"
    assert sync_body["sample_count"] == 4
    assert sync_body["accuracy"] == 0.75
    assert sync_body["precision"] == 0.5
    assert sync_body["recall"] == 1.0

    # 2. Async worker evaluation via queue_service (memory mode)
    test_session_factory = async_sessionmaker(
        bind=db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    queue_service.set_force_memory(True)
    queue_service.set_session_factory(test_session_factory)
    await queue_service._memory_queue.clear()

    async_resp = await async_client.post(
        f"/api/v1/models/{model.id}/performance/evaluate?async_mode=true"
    )
    assert async_resp.status_code == 200
    async_body = async_resp.json()
    assert async_body["status"] == "QUEUED"
    job_id = async_body["job_id"]

    # Wait briefly for in-memory worker task completion
    job_status = None
    for _ in range(30):
        job_status = await queue_service.get_job(job_id)
        if job_status and job_status.status in ("complete", "failed"):
            break
        await asyncio.sleep(0.05)

    assert job_status is not None
    assert job_status.status == "complete"
    assert job_status.result["accuracy"] == 0.75
    assert job_status.result["sample_count"] == 4


# ------------------------------------------------------------------------------
# 4. Strict Metric Separation Test (Offline vs Online vs Delayed Ground-Truth)
# ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strict_separation_of_offline_online_and_delayed_metrics(
    db_session: AsyncSession,
) -> None:
    """Verify delayed ground-truth evaluation never mixes with or mutates offline/online metrics."""
    _, version = await _create_model_and_version(db_session, "metric-separation-model")
    now = datetime.now(timezone.utc)

    # 1. Seed static pre-deployment offline metrics (high benchmark accuracy = 0.98)
    offline = OfflineMetrics(
        model_version_id=version.id,
        dataset_split="test",
        accuracy=0.98,
        precision=0.97,
        recall=0.96,
        f1_score=0.965,
        roc_auc=0.99,
    )
    # 2. Seed separate online_metrics record
    online = OnlineMetrics(
        model_version_id=version.id,
        window_start=now - timedelta(hours=1),
        window_end=now,
        sample_count=500,
        metrics_payload={"avg_confidence": 0.94, "predicted_positive_rate": 0.12},
    )
    db_session.add_all([offline, online])
    await db_session.flush()

    # 3. Simulate degraded production ground-truth (accuracy = 0.60, 3 of 5 correct)
    pairs = [(1, 1), (1, 0), (0, 1), (0, 0), (0, 0)]
    for i, (pred, actual) in enumerate(pairs):
        inf = InferenceRequest(
            id=uuid.uuid4(),
            request_id=f"req-sep-{i}",
            model_version_id=version.id,
            input_features={"x1": float(i)},
            prediction={"value": pred, "confidence": 0.95},
            latency_ms=3.5,
            cached=False,
            created_at=now - timedelta(hours=5 - i),
        )
        db_session.add(inf)
        await db_session.flush()
        db_session.add(
            GroundTruth(
                inference_request_id=inf.id,
                actual_label={"value": actual},
            )
        )
    await db_session.commit()

    # 4. Run delayed ground-truth evaluation worker
    test_session_factory = async_sessionmaker(
        bind=db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    await evaluate_ground_truth_task(
        ctx={"session_factory": test_session_factory},
        model_version_id=str(version.id),
    )

    # 5. Assert strict separation across all three tables
    # (a) Offline metrics remain completely untouched (0.98)
    offline_after = (
        await db_session.execute(
            select(OfflineMetrics).where(OfflineMetrics.model_version_id == version.id)
        )
    ).scalar_one()
    assert offline_after.accuracy == 0.98

    # (b) Online metrics remain completely untouched (still 1 row, no accuracy overwritten)
    online_rows = (
        (
            await db_session.execute(
                select(OnlineMetrics).where(OnlineMetrics.model_version_id == version.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(online_rows) == 1
    assert online_rows[0].accuracy is None

    # (c) Delayed ground-truth performance is isolated in perf_metrics (0.60)
    #     and NEVER mixes operational serving telemetry (latency/cache/error fields remain None)
    perf_rows = (
        (
            await db_session.execute(
                select(PerfMetrics).where(PerfMetrics.model_version_id == version.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(perf_rows) == 1
    perf = perf_rows[0]
    assert perf.sample_count == 5
    assert perf.accuracy == 0.6
    assert perf.precision == 0.5
    assert perf.recall == 0.5
    assert perf.f1_score == 0.5
    assert perf.confusion_matrix == {
        "labels": [0, 1],
        "matrix": [[2, 1], [1, 1]],
        "tp": 1,
        "tn": 2,
        "fp": 1,
        "fn": 1,
        "averaging": "binary",
    }
    assert perf.total_requests is None
    assert perf.cache_hits is None
    assert perf.p50_latency_ms is None
    assert perf.p95_latency_ms is None
    assert perf.p99_latency_ms is None
    assert perf.error_count is None
