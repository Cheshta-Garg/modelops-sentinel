import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.base import Base
from app.models.deployment import Deployment
from app.models.enums import (
    ArtifactFormat,
    DeploymentStatus,
    DeploymentStrategy,
    DriftMetric,
    ModelStatus,
    UserRole,
)
from app.models.inference import GroundTruth, InferenceRequest
from app.models.metrics import (
    OfflineMetrics,
    OnlineMetrics,
    PerfMetrics,
    ReferenceStats,
)
from app.models.model import Model, ModelVersion
from app.models.monitoring import AuditLog, DriftEvent
from app.models.user import User

# In-memory async SQLite database for isolation testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Provide an isolated in-memory async SQLite engine with all tables created."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Provide an async session scoped per test."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session


@pytest.mark.asyncio
async def test_all_12_tables_created(test_engine: AsyncEngine) -> None:
    """Verify that all 12 platform tables exist in Base.metadata."""
    expected_tables = {
        "users",
        "models",
        "model_versions",
        "offline_metrics",
        "reference_stats",
        "deployments",
        "inference_requests",
        "ground_truth",
        "online_metrics",
        "perf_metrics",
        "drift_events",
        "audit_logs",
    }
    actual_tables = set(Base.metadata.tables.keys())
    assert expected_tables.issubset(actual_tables), (
        f"Missing tables: {expected_tables - actual_tables}"
    )


@pytest.mark.asyncio
async def test_create_user_and_model(db_session: AsyncSession) -> None:
    """Test user creation and model association."""
    user = User(
        email="lead.mle@company.com",
        hashed_password="secure_hashed_password_example",
        full_name="Lead ML Engineer",
        role=UserRole.ADMIN,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    assert user.id is not None
    assert user.role == UserRole.ADMIN

    model = Model(
        name="fraud_detection_xgboost",
        description="Transaction fraud classifier",
        task_type="binary_classification",
        created_by_id=user.id,
    )
    db_session.add(model)
    await db_session.commit()
    await db_session.refresh(model)

    assert model.name == "fraud_detection_xgboost"
    assert model.created_by_id == user.id


@pytest.mark.asyncio
async def test_user_unique_email_constraint(db_session: AsyncSession) -> None:
    """Verify duplicate user email violates unique constraint."""
    user1 = User(
        email="duplicate@company.com",
        hashed_password="hash1",
        full_name="User One",
    )
    user2 = User(
        email="duplicate@company.com",
        hashed_password="hash2",
        full_name="User Two",
    )
    db_session.add(user1)
    await db_session.commit()

    db_session.add(user2)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_model_version_unique_tag_constraint(db_session: AsyncSession) -> None:
    """Verify unique constraint on (model_id, version_tag)."""
    model = Model(name="churn_nn", task_type="classification")
    db_session.add(model)
    await db_session.commit()

    v1 = ModelVersion(
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.DRAFT,
        artifact_path="/artifacts/churn/v1/model.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        input_schema={"features": ["amount", "tenure"]},
        output_schema={"prediction": "int", "probability": "float"},
    )
    db_session.add(v1)
    await db_session.commit()

    v1_dup = ModelVersion(
        model_id=model.id,
        version_tag="v1.0.0",  # Duplicate tag for same model
        status=ModelStatus.STAGING,
        artifact_path="/artifacts/churn/v1_dup/model.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        input_schema={"features": ["amount", "tenure"]},
        output_schema={"prediction": "int", "probability": "float"},
    )
    db_session.add(v1_dup)
    with pytest.raises(IntegrityError):
        await db_session.commit()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_offline_metrics_and_reference_stats(db_session: AsyncSession) -> None:
    """Verify offline metrics and reference stats persistence and 1-to-1 relationship."""
    model = Model(name="credit_risk", task_type="classification")
    db_session.add(model)
    await db_session.commit()

    version = ModelVersion(
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.STAGING,
        artifact_path="/models/credit/v1/model.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="abc123hash",
        input_schema={"features": ["income", "score"]},
        output_schema={"score": "float"},
    )
    db_session.add(version)
    await db_session.commit()

    offline_metrics = OfflineMetrics(
        model_version_id=version.id,
        dataset_split="test",
        accuracy=0.942,
        precision=0.915,
        recall=0.887,
        f1_score=0.901,
        roc_auc=0.973,
        custom_metrics={"pr_auc": 0.89},
    )
    db_session.add(offline_metrics)

    ref_stats = ReferenceStats(
        model_version_id=version.id,
        num_samples=10000,
        feature_stats={
            "income": {"mean": 65000.0, "std": 15000.0, "p50": 62000.0},
            "score": {"mean": 720.0, "std": 45.0, "p50": 725.0},
        },
        prediction_stats={"class_0": 0.88, "class_1": 0.12},
    )
    db_session.add(ref_stats)
    await db_session.commit()

    # Query back and verify
    stmt = select(ModelVersion).where(ModelVersion.id == version.id)
    result = await db_session.execute(stmt)
    fetched_version = result.scalar_one()

    assert fetched_version.offline_metrics.accuracy == 0.942
    assert fetched_version.reference_stats.num_samples == 10000


@pytest.mark.asyncio
async def test_deployment_canary_routing(db_session: AsyncSession) -> None:
    """Verify deployment with primary and canary versions."""
    model = Model(name="sentiment_classifier", task_type="nlp")
    db_session.add(model)
    await db_session.commit()

    v1 = ModelVersion(
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.ACTIVE,
        artifact_path="/models/nlp/v1/model.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="hash_v1",
        input_schema={"text": "string"},
        output_schema={"sentiment": "string"},
    )
    v2 = ModelVersion(
        model_id=model.id,
        version_tag="v2.0.0",
        status=ModelStatus.STAGING,
        artifact_path="/models/nlp/v2/model.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="hash_v2",
        input_schema={"text": "string"},
        output_schema={"sentiment": "string"},
    )
    db_session.add_all([v1, v2])
    await db_session.commit()

    deployment = Deployment(
        model_id=model.id,
        environment="production",
        status=DeploymentStatus.ACTIVE,
        strategy=DeploymentStrategy.CANARY,
        primary_version_id=v1.id,
        canary_version_id=v2.id,
        traffic_split=90.0,  # 90% v1, 10% v2
    )
    db_session.add(deployment)
    await db_session.commit()
    await db_session.refresh(deployment)

    assert deployment.traffic_split == 90.0
    assert deployment.primary_version.version_tag == "v1.0.0"
    assert deployment.canary_version.version_tag == "v2.0.0"


@pytest.mark.asyncio
async def test_inference_request_and_ground_truth(db_session: AsyncSession) -> None:
    """Test telemetry logging of inference request and subsequent ground truth matching."""
    model = Model(name="recsys", task_type="ranking")
    db_session.add(model)
    await db_session.commit()

    version = ModelVersion(
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.ACTIVE,
        artifact_path="/models/rec/v1.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="hash_rec",
        input_schema={"user_id": "int"},
        output_schema={"item_id": "int"},
    )
    db_session.add(version)
    await db_session.commit()

    req = InferenceRequest(
        request_id="req-trace-abc-123",
        model_version_id=version.id,
        input_features={"user_id": 42},
        prediction={"recommended_item": 109, "score": 0.95},
        latency_ms=12.4,
        cached=False,
    )
    db_session.add(req)
    await db_session.commit()

    gt = GroundTruth(
        inference_request_id=req.id,
        actual_label={"converted": True, "clicked_item": 109},
    )
    db_session.add(gt)
    await db_session.commit()
    await db_session.refresh(req)

    assert req.ground_truth is not None
    assert req.ground_truth.actual_label["converted"] is True


@pytest.mark.asyncio
async def test_online_and_perf_metrics(db_session: AsyncSession) -> None:
    """Test online window accuracy metrics and perf latency tracking."""
    model = Model(name="forecasting", task_type="timeseries")
    db_session.add(model)
    await db_session.commit()

    version = ModelVersion(
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.ACTIVE,
        artifact_path="/models/ts/v1.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="hash_ts",
        input_schema={},
        output_schema={},
    )
    db_session.add(version)
    await db_session.commit()

    now = datetime.now(timezone.utc)
    online_metric = OnlineMetrics(
        model_version_id=version.id,
        window_start=now,
        window_end=now,
        sample_count=500,
        accuracy=0.89,
        f1_score=0.87,
    )
    perf_metric = PerfMetrics(
        model_version_id=version.id,
        window_start=now,
        window_end=now,
        total_requests=1000,
        cache_hits=400,
        p50_latency_ms=8.5,
        p95_latency_ms=24.1,
        p99_latency_ms=58.3,
        error_count=1,
    )
    db_session.add_all([online_metric, perf_metric])
    await db_session.commit()

    assert online_metric.accuracy == 0.89
    assert perf_metric.p99_latency_ms == 58.3


@pytest.mark.asyncio
async def test_drift_event_and_audit_log(db_session: AsyncSession) -> None:
    """Test recording of drift events and compliance audit logs."""
    user = User(
        email="auditor@company.com",
        hashed_password="pw",
        full_name="Compliance Auditor",
        role=UserRole.ADMIN,
    )
    model = Model(name="loan_underwriter", task_type="classification")
    db_session.add_all([user, model])
    await db_session.commit()

    version = ModelVersion(
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.ACTIVE,
        artifact_path="/path.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="hash",
        input_schema={},
        output_schema={},
    )
    db_session.add(version)
    await db_session.commit()

    now = datetime.now(timezone.utc)
    drift = DriftEvent(
        model_version_id=version.id,
        window_start=now,
        window_end=now,
        drift_detected=True,
        metric_used=DriftMetric.PSI,
        threshold=0.25,
        overall_drift_score=0.34,
        feature_drift_details={"credit_score": {"psi": 0.38, "drift": True}},
    )
    audit = AuditLog(
        user_id=user.id,
        action="CANARY_ROLLBACK",
        entity_type="deployment",
        entity_id=str(uuid.uuid4()),
        payload_before={"traffic_split": 90.0},
        payload_after={"traffic_split": 100.0, "reason": "Drift and latency breach"},
    )
    db_session.add_all([drift, audit])
    await db_session.commit()

    assert drift.drift_detected is True
    assert audit.action == "CANARY_ROLLBACK"
    assert audit.user.email == "auditor@company.com"


@pytest.mark.asyncio
async def test_cascade_delete_model_removes_children(db_session: AsyncSession) -> None:
    """Test that deleting a Model cascades to ModelVersion and related metrics."""
    model = Model(name="temp_model", task_type="classification")
    db_session.add(model)
    await db_session.commit()

    version = ModelVersion(
        model_id=model.id,
        version_tag="v1.0.0",
        status=ModelStatus.DRAFT,
        artifact_path="/temp.joblib",
        artifact_format=ArtifactFormat.JOBLIB,
        artifact_hash_sha256="hash",
        input_schema={},
        output_schema={},
    )
    db_session.add(version)
    await db_session.commit()

    offline_metric = OfflineMetrics(
        model_version_id=version.id,
        dataset_split="test",
        accuracy=0.85,
    )
    db_session.add(offline_metric)
    await db_session.commit()

    # Verify rows exist
    version_id = version.id
    metric_id = offline_metric.id

    # Delete the parent Model
    await db_session.delete(model)
    await db_session.commit()

    # Query children - they should be gone
    res_version = await db_session.execute(
        select(ModelVersion).where(ModelVersion.id == version_id)
    )
    assert res_version.scalar_one_or_none() is None

    res_metric = await db_session.execute(
        select(OfflineMetrics).where(OfflineMetrics.id == metric_id)
    )
    assert res_metric.scalar_one_or_none() is None
