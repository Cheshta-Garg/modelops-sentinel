"""Comprehensive Stage 7 Manual Verification Script for ModelOps Sentinel.

This script exercises the complete telemetry pipeline end-to-end:
1. HTTP request & error tracking
2. Model inference latency and model/version usage
3. Inference result cache hits and misses
4. Asynchronous background jobs
5. Prometheus GET /metrics scrape format & metric presence
"""

import asyncio
import io
import sys
import uuid
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from sklearn.linear_model import LogisticRegression
from starlette.testclient import TestClient

from app.core.security import hash_password
from app.db.base import Base
from app.db.session import AsyncSessionLocal, async_engine
from app.main import app
from app.models.enums import ModelStatus, UserRole
from app.models.model import Model, ModelVersion
from app.models.user import User
from app.storage.artifact_store import LocalArtifactStore


def print_section(title: str) -> None:
    print(f"\n{'=' * 75}\n  {title}\n{'=' * 75}")


async def setup_test_fixtures() -> tuple[str, str]:
    """Seed a test user and trained model in the database."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    storage = LocalArtifactStore()

    # Train dummy logistic regression model
    clf = LogisticRegression()
    clf.fit([[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]], [0, 1])
    buf = io.BytesIO()
    joblib.dump(clf, buf)
    from app.storage.artifact_store import compute_sha256

    content_bytes = buf.getvalue()
    model_name = f"iris_sentinel_stage7_{uuid.uuid4().hex[:4]}"
    rel_path = await storage.save(model_name, "v1.0.0", "model.joblib", content_bytes)
    file_hash = compute_sha256(content_bytes)

    model_id = uuid.uuid4()
    version_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        # Create engineer user
        user = User(
            id=uuid.uuid4(),
            email=f"stage7_engineer_{uuid.uuid4().hex[:6]}@modelops.internal",
            full_name="Stage 7 Engineer",
            hashed_password=hash_password("Password123!"),
            role=UserRole.ENGINEER,
            is_active=True,
        )
        session.add(user)

        # Create model and version
        model = Model(
            id=model_id,
            name=model_name,
            description="Observability verification test model",
            created_by_id=user.id,
        )
        session.add(model)
        await session.flush()

        version = ModelVersion(
            id=version_id,
            model_id=model.id,
            version_tag="v1.0.0",
            status=ModelStatus.ACTIVE,
            artifact_path=rel_path,
            artifact_hash_sha256=file_hash,
            input_schema={
                "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            },
            output_schema={"prediction": "int"},
            created_by_id=user.id,
        )
        session.add(version)
        await session.commit()

        # Login to get JWT
        client = TestClient(app)
        login_resp = client.post(
            "/api/v1/auth/login", json={"email": user.email, "password": "Password123!"}
        )
        token = login_resp.json()["access_token"]
        return str(model_id), token


async def main() -> None:
    print_section("STAGE 7: PROMETHEUS METRICS & OBSERVABILITY VERIFICATION")

    from app.services.queue import queue_service

    queue_service.set_force_memory(True)
    queue_service.set_session_factory(AsyncSessionLocal)

    model_id, token = await setup_test_fixtures()
    client = TestClient(app)
    auth_headers = {"Authorization": f"Bearer {token}"}

    # 1. Exercise HTTP requests & error tracking
    print("\n[Step 1] Exercising HTTP Traffic (Success & Error paths)...")
    r1 = client.get("/api/v1/health")
    print(f"  GET /api/v1/health -> {r1.status_code}")

    r2 = client.get("/api/v1/non_existent_route")
    print(f"  GET /api/v1/non_existent_route -> {r2.status_code} (Expected 404 for error tracking)")

    # 2. Exercise Synchronous Inference & Cache Hit/Miss
    print("\n[Step 2] Exercising Inference Latency, Model Usage & Cache Hit/Miss...")
    payload = {
        "features": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
    }
    # First call -> Cache MISS, computes inference
    inf1 = client.post(f"/api/v1/models/{model_id}/predict", json=payload, headers=auth_headers)
    print(
        f"  First prediction (Cache MISS): status={inf1.status_code}, cached={inf1.json().get('cached')}, latency={inf1.json().get('latency_ms')}ms"
    )

    # Second identical call -> Cache HIT
    inf2 = client.post(f"/api/v1/models/{model_id}/predict", json=payload, headers=auth_headers)
    print(
        f"  Second prediction (Cache HIT): status={inf2.status_code}, cached={inf2.json().get('cached')}, latency={inf2.json().get('latency_ms')}ms"
    )

    # 3. Exercise Asynchronous Inference Job
    print("\n[Step 3] Exercising Async Background Job Tracking...")
    async_resp = client.post(
        f"/api/v1/models/{model_id}/predict/async", json=payload, headers=auth_headers
    )
    print(f"  POST /api/v1/models/{model_id}/predict/async -> {async_resp.status_code}")
    job_id = async_resp.json().get("job_id")
    print(f"  Enqueued job_id: {job_id}")

    # Wait for memory worker to complete
    await asyncio.sleep(0.3)
    job_status = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
    print(
        f"  GET /api/v1/jobs/{job_id} -> {job_status.status_code}, status={job_status.json().get('status')}"
    )

    # 4. Scrape GET /metrics
    print("\n[Step 4] Scraping GET /metrics (Prometheus exposition format)...")
    metrics_resp = client.get("/metrics")
    print(
        f"  GET /metrics -> {metrics_resp.status_code} {metrics_resp.headers.get('content-type')}"
    )
    metrics_text = metrics_resp.text

    # 5. Validate Required Metrics
    required_metrics = [
        ("request count", "modelops_http_requests_total"),
        ("error count", "modelops_http_errors_total"),
        ("request latency", "modelops_http_request_duration_seconds"),
        ("inference latency", "modelops_inference_duration_seconds"),
        ("model/version usage", "modelops_model_inferences_total"),
        ("async job count", "modelops_async_jobs_total"),
        ("cache hit/miss", "modelops_inference_cache_requests_total"),
    ]

    print("\n[Step 5] Validating Presence of All Required Prometheus Metrics:")
    all_passed = True
    for label, metric_name in required_metrics:
        present = metric_name in metrics_text
        status_icon = "PASS" if present else "FAIL"
        if not present:
            all_passed = False
        print(f"  [{status_icon:^4}] {label:<22} -> {metric_name}")

    # Print metric sample lines
    print("\n[Step 6] Sample Scraped Metric Lines from /metrics:")
    for line in metrics_text.splitlines():
        if any(m[1] in line for m in required_metrics) and not line.startswith("#"):
            print(f"    {line}")

    print_section(
        "STAGE 7 VERIFICATION SUMMARY: "
        + ("ALL METRICS OPERATIONAL" if all_passed else "FAILURES DETECTED")
    )


if __name__ == "__main__":
    asyncio.run(main())
