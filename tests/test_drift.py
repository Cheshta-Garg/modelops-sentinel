import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ArtifactFormat, ModelStatus, UserRole
from app.models.inference import InferenceRequest
from app.models.model import Model, ModelVersion
from app.models.monitoring import DriftEvent
from app.models.user import User
from app.services.drift import (
    DriftService,
    calculate_ks_test,
    calculate_psi,
    classify_ks_severity,
    classify_psi_severity,
    extract_reference_statistics,
)
from app.worker.tasks import evaluate_drift_task


def test_psi_identical_distribution():
    """Verify that samples from identical distributions produce negligible PSI (< 0.1)."""
    np.random.seed(42)
    reference = np.random.normal(loc=100.0, scale=15.0, size=500).tolist()
    target = np.random.normal(loc=100.0, scale=15.0, size=500).tolist()

    psi_score, details = calculate_psi(reference, target, num_bins=10)
    assert psi_score < 0.10, f"Expected low PSI, got {psi_score}"
    assert classify_psi_severity(psi_score) == "NONE"
    assert "num_bins" in details


def test_psi_shifted_distribution():
    """Verify that a significant distribution shift triggers HIGH severity PSI (> 0.25)."""
    np.random.seed(42)
    reference = np.random.normal(loc=100.0, scale=15.0, size=500).tolist()
    target = np.random.normal(loc=130.0, scale=15.0, size=500).tolist()  # +2 sigma shift

    psi_score, details = calculate_psi(reference, target, num_bins=10)
    assert psi_score > 0.25, f"Expected high PSI, got {psi_score}"
    assert classify_psi_severity(psi_score) == "HIGH"


def test_ks_identical_distribution():
    """Verify that identical distributions yield small KS statistic and non-significant p-value."""
    np.random.seed(42)
    reference = np.random.normal(loc=0.0, scale=1.0, size=300).tolist()
    target = np.random.normal(loc=0.0, scale=1.0, size=300).tolist()

    stat, p_val, detected = calculate_ks_test(reference, target)
    assert stat < 0.15
    assert p_val > 0.05
    assert detected is False
    assert classify_ks_severity(stat, p_val) == "NONE"


def test_ks_shifted_distribution():
    """Verify that a shifted distribution yields high KS statistic and p-value < 0.001."""
    np.random.seed(42)
    reference = np.random.normal(loc=0.0, scale=1.0, size=300).tolist()
    target = np.random.normal(loc=1.5, scale=1.0, size=300).tolist()

    stat, p_val, detected = calculate_ks_test(reference, target)
    assert stat > 0.25
    assert p_val < 0.001
    assert detected is True
    assert classify_ks_severity(stat, p_val) == "HIGH"


def test_extract_reference_statistics():
    """Verify multi-feature reference profile calculation."""
    np.random.seed(42)
    data = {
        "feat_a": np.random.uniform(10.0, 50.0, size=100).tolist(),
        "feat_b": np.random.uniform(20.0, 100.0, size=100).tolist(),
    }
    ref_stats = extract_reference_statistics(data, feature_names=["feat_a", "feat_b"])

    assert "feat_a" in ref_stats
    assert "feat_b" in ref_stats
    assert ref_stats["feat_a"]["count"] == 100
    assert "mean" in ref_stats["feat_a"]
    assert "std" in ref_stats["feat_a"]
    assert "bin_edges" in ref_stats["feat_a"]
    assert len(ref_stats["feat_a"]["sample_values"]) > 0


@pytest.mark.asyncio
async def test_drift_evaluation_and_events_recording(db_session: AsyncSession):
    """Verify DriftService evaluates prediction logs against reference data and stores drift_events."""
    user = User(
        id=uuid.uuid4(),
        email="drift_tester@modelops.test",
        full_name="Drift Tester",
        hashed_password="mock",
        role=UserRole.ENGINEER,
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    model = Model(
        id=uuid.uuid4(),
        name="drift-model-test",
        description="Testing drift recording",
        task_type="binary_classification",
        created_by_id=user.id,
    )
    db_session.add(model)
    await db_session.flush()

    version = ModelVersion(
        id=uuid.uuid4(),
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.ACTIVE,
        artifact_path="/mock/model.pkl",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="mock_sha256_hash",
        input_schema={"type": "object"},
        output_schema={"type": "number"},
        created_by_id=user.id,
    )
    db_session.add(version)
    await db_session.flush()

    # Create reference statistics
    np.random.seed(42)
    ref_dict = {
        "income": np.random.normal(50000, 5000, size=200).tolist(),
        "score": np.random.normal(700, 30, size=200).tolist(),
    }
    await DriftService.save_reference_stats(
        session=db_session,
        model_version_id=version.id,
        feature_data=ref_dict,
        feature_names=["income", "score"],
    )

    # Insert shifted incoming predictions (income shifted upward, score unchanged)
    for _ in range(60):
        req = InferenceRequest(
            id=uuid.uuid4(),
            request_id=str(uuid.uuid4()),
            model_version_id=version.id,
            input_features={
                "income": float(np.random.normal(85000, 5000)),  # heavy drift
                "score": float(np.random.normal(700, 30)),  # no drift
            },
            prediction={"score": 0.15},
            latency_ms=1.2,
            cached=False,
        )
        db_session.add(req)
    await db_session.commit()

    # Evaluate drift using PSI
    events = await DriftService.evaluate_drift(
        session=db_session,
        model_version_id=version.id,
        method="PSI",
        window_size=60,
        threshold=0.25,
    )

    assert len(events) >= 2
    income_ev = next(e for e in events if e.feature == "income")
    score_ev = next(e for e in events if e.feature == "score")

    assert income_ev.drift_detected is True
    assert income_ev.severity in ["MEDIUM", "HIGH"]
    assert score_ev.drift_detected is False
    assert score_ev.severity in ["NONE", "MEDIUM"]

    # Verify drift_events were persisted with all required columns
    report = await DriftService.get_drift_events_for_model(
        session=db_session,
        model_id=model.id,
    )
    assert len(report["events"]) >= 2
    for ev in report["events"]:
        assert ev["feature"] in ["income", "score"]
        assert ev["method"] == "PSI"
        assert isinstance(ev["score"], float)
        assert ev["threshold"] == 0.25
        assert ev["severity"] in ["NONE", "LOW", "MEDIUM", "HIGH"]
        assert ev["detected_at"] is not None


@pytest.mark.asyncio
async def test_get_model_drift_api(async_client: AsyncClient, db_session: AsyncSession):
    """Verify GET /api/v1/models/{id}/drift returns the model drift status, summary, and events."""
    user = User(
        id=uuid.uuid4(),
        email="api_drift_user@modelops.test",
        full_name="API Drift User",
        hashed_password="mock",
        role=UserRole.ADMIN,
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    model = Model(
        id=uuid.uuid4(),
        name="api-drift-model",
        task_type="binary_classification",
        created_by_id=user.id,
    )
    db_session.add(model)
    await db_session.flush()

    version = ModelVersion(
        id=uuid.uuid4(),
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.ACTIVE,
        artifact_path="/mock/model.pkl",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="mock_hash_123",
        input_schema={"type": "object"},
        output_schema={"type": "number"},
        created_by_id=user.id,
    )
    db_session.add(version)
    await db_session.flush()

    # Create reference statistics
    ref_dict = {"age": [float(30 + i % 10) for i in range(100)]}
    await DriftService.save_reference_stats(
        session=db_session,
        model_version_id=version.id,
        feature_data=ref_dict,
        feature_names=["age"],
    )

    # Add a mock drift event
    drift_event = DriftEvent(
        id=uuid.uuid4(),
        model_version_id=version.id,
        feature="age",
        method="KS_TEST",
        score=0.45,
        threshold=0.25,
        severity="HIGH",
        detected_at=datetime.now(timezone.utc),
        drift_detected=True,
    )
    db_session.add(drift_event)
    await db_session.commit()

    # Call GET /api/v1/models/{id}/drift
    response = await async_client.get(f"/api/v1/models/{model.id}/drift")
    assert response.status_code == 200

    body = response.json()
    assert body["model_id"] == str(model.id)
    assert body["version_tag"] == "v1.0.0"
    assert body["drift_detected"] is True
    assert len(body["events"]) >= 1
    first_ev = body["events"][0]
    assert first_ev["feature"] == "age"
    assert first_ev["method"] == "KS_TEST"
    assert first_ev["score"] == 0.45
    assert first_ev["threshold"] == 0.25
    assert first_ev["severity"] == "HIGH"
    assert first_ev["detected_at"] is not None
    assert body["summary"]["drifted_features_count"] >= 1


@pytest.mark.asyncio
async def test_worker_evaluate_drift_task(db_session: AsyncSession):
    """Verify background worker task can evaluate drift asynchronously."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    user = User(
        id=uuid.uuid4(),
        email="worker_user@modelops.test",
        full_name="Worker User",
        hashed_password="mock",
        role=UserRole.ADMIN,
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    model = Model(
        id=uuid.uuid4(),
        name="worker-drift-model",
        task_type="binary_classification",
        created_by_id=user.id,
    )
    db_session.add(model)
    await db_session.flush()

    version = ModelVersion(
        id=uuid.uuid4(),
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.ACTIVE,
        artifact_path="/mock/model.pkl",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="mock_hash_456",
        input_schema={"type": "object"},
        output_schema={"type": "number"},
        created_by_id=user.id,
    )
    db_session.add(version)
    await db_session.flush()

    # Save reference stats
    ref_dict = {"x": [float(i) for i in range(50)]}
    await DriftService.save_reference_stats(
        session=db_session,
        model_version_id=version.id,
        feature_data=ref_dict,
        feature_names=["x"],
    )

    # Insert production predictions
    for i in range(40):
        db_session.add(
            InferenceRequest(
                id=uuid.uuid4(),
                request_id=str(uuid.uuid4()),
                model_version_id=version.id,
                input_features={"x": float(i + 100)},  # shifted
                prediction={"score": 1.0},
                latency_ms=1.5,
                cached=False,
            )
        )
    await db_session.commit()

    # Pass in-memory test session factory via ctx
    test_session_factory = async_sessionmaker(
        bind=db_session.bind,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    ctx = {"session_factory": test_session_factory}

    result = await evaluate_drift_task(
        ctx,
        model_version_id=str(version.id),
        method="KS_TEST",
        threshold=0.25,
        window_size=50,
    )

    assert result["status"] == "COMPLETED"
    assert result["model_version_id"] == str(version.id)
    assert result["drift_detected"] is True
    assert result["drifted_features_count"] >= 1
