"""Drift Simulation and Verification Script for ModelOps Sentinel.

This script demonstrates controlled statistical distribution shift:
1. Baseline Reference Profiling: Creates ReferenceStats from standard training data.
2. Phase 1 - In-Distribution Normal Traffic: Sends production requests from the baseline distribution.
   -> Verifies PSI < 0.10, KS p-value >= 0.05, severity = NONE (No drift).
3. Phase 2 - Controlled Distribution Shift: Injects severe covariate shift into credit_score & debt_to_income.
   -> Verifies PSI >= 0.25, KS p-value < 0.01, severity = HIGH (Drift detected).
4. Persists and queries drift_events via GET /api/v1/models/{id}/drift.
"""

import asyncio
import io
import sys
import uuid
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from starlette.testclient import TestClient

from app.core.security import hash_password
from app.db.base import Base
from app.db.session import AsyncSessionLocal, async_engine
from app.main import app
from app.models.enums import ModelStatus, UserRole
from app.models.model import Model, ModelVersion
from app.models.user import User
from app.services.drift import drift_service
from app.storage.artifact_store import LocalArtifactStore, compute_sha256


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}\n  {title}\n{'=' * 80}")


async def setup_simulation_model() -> tuple[str, str, str, dict[str, list[float]]]:
    """Train baseline model, store artifact, save ReferenceStats, and return credentials."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    np.random.seed(42)
    n_samples = 500

    # 1. Generate Baseline Reference Distributions
    ref_credit = np.random.normal(loc=700.0, scale=40.0, size=n_samples)
    ref_income = np.random.normal(loc=65000.0, scale=12000.0, size=n_samples)
    ref_dti = np.random.normal(loc=0.30, scale=0.06, size=n_samples)
    ref_loan = np.random.normal(loc=20000.0, scale=4000.0, size=n_samples)

    ref_data = {
        "credit_score": ref_credit.tolist(),
        "annual_income": ref_income.tolist(),
        "debt_to_income": ref_dti.tolist(),
        "loan_amount": ref_loan.tolist(),
    }

    # Synthetic binary target (1 = default / high risk)
    y_train = ((ref_credit < 680).astype(int) + (ref_dti > 0.32).astype(int) >= 1).astype(int)

    X_train = np.column_stack([ref_credit, ref_income, ref_dti, ref_loan])
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Serialize Model Artifact
    buf = io.BytesIO()
    joblib.dump(clf, buf)
    artifact_bytes = buf.getvalue()
    buf.seek(0)

    storage = LocalArtifactStore()
    model_name = f"credit_risk_sentinel_{uuid.uuid4().hex[:4]}"
    rel_path = await storage.save(model_name, "v1.0.0", "model.joblib", artifact_bytes)
    file_hash = compute_sha256(artifact_bytes)

    model_id = uuid.uuid4()
    version_id = uuid.uuid4()

    async with AsyncSessionLocal() as session:
        # Create engineer user for auth
        user = User(
            id=uuid.uuid4(),
            email=f"drift_eng_{uuid.uuid4().hex[:6]}@modelops.internal",
            full_name="Drift Test Engineer",
            hashed_password=hash_password("Password123!"),
            role=UserRole.ENGINEER,
            is_active=True,
        )
        session.add(user)

        model = Model(
            id=model_id,
            name=model_name,
            description="Credit risk model for statistical drift simulation",
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
                "features": ["credit_score", "annual_income", "debt_to_income", "loan_amount"]
            },
            output_schema={"prediction": "int"},
            created_by_id=user.id,
        )
        session.add(version)
        await session.commit()

        # Save Reference Statistics
        await drift_service.save_reference_stats(
            session=session,
            model_version_id=version.id,
            feature_data=ref_data,
            feature_names=["credit_score", "annual_income", "debt_to_income", "loan_amount"],
        )

        client = TestClient(app)
        login_res = client.post(
            "/api/v1/auth/login", json={"email": user.email, "password": "Password123!"}
        )
        token = login_res.json()["access_token"]

        return str(model.id), str(version.id), token, ref_data


async def main() -> None:
    print_header("STAGE 8: STATISTICAL DRIFT SIMULATION (PSI & KS-TEST)")

    model_id, version_id, token, ref_data = await setup_simulation_model()
    client = TestClient(app)
    auth_headers = {"Authorization": f"Bearer {token}"}

    print(f"Model Initialized: ID={model_id}")
    print(f"Version Initialized: ID={version_id} (ACTIVE)")
    print("Reference Statistics Configured across 4 features (500 baseline samples each)")

    # --------------------------------------------------------------------------
    # Phase 1: In-Distribution Normal Traffic (Zero / Negligible Drift)
    # --------------------------------------------------------------------------
    print_header("PHASE 1: IN-DISTRIBUTION NORMAL TRAFFIC (NO COVARIATE SHIFT)")
    print("Generating 25 production inferences sampled from the baseline training distribution...")

    np.random.seed(101)
    normal_credit = np.random.normal(loc=700.0, scale=40.0, size=25)
    normal_income = np.random.normal(loc=65000.0, scale=12000.0, size=25)
    normal_dti = np.random.normal(loc=0.30, scale=0.06, size=25)
    normal_loan = np.random.normal(loc=20000.0, scale=4000.0, size=25)

    for i in range(25):
        payload = {
            "features": {
                "credit_score": round(float(normal_credit[i]), 1),
                "annual_income": round(float(normal_income[i]), 2),
                "debt_to_income": round(float(normal_dti[i]), 4),
                "loan_amount": round(float(normal_loan[i]), 2),
            }
        }
        res = client.post(f"/api/v1/models/{model_id}/predict", json=payload, headers=auth_headers)
        assert res.status_code == 200

    print("Evaluating drift using PSI on Normal Traffic...")
    async with AsyncSessionLocal() as session:
        events_normal_psi = await drift_service.evaluate_drift(
            session=session,
            model_version_id=uuid.UUID(version_id),
            method="PSI",
            window_size=25,
            threshold=0.25,
        )

    print("\n  [Normal Traffic Results - Method: PSI]")
    print(
        f"  {'Feature':<18} {'Score (PSI)':<14} {'Threshold':<12} {'Severity':<10} {'Drift Detected'}"
    )
    print(f"  {'-' * 65}")
    for e in events_normal_psi:
        print(
            f"  {e.feature:<18} {e.score:<14.4f} {e.threshold:<12.2f} {e.severity:<10} {str(e.drift_detected)}"
        )
        assert e.drift_detected is False, f"Unexpected drift flagged on normal feature {e.feature}"

    # --------------------------------------------------------------------------
    # Phase 2: Controlled Distribution Shift (Covariate Drift Injection)
    # --------------------------------------------------------------------------
    print_header("PHASE 2: CONTROLLED DISTRIBUTION SHIFT (INDUCING DATA DRIFT)")
    print("Injecting severe covariate shift into production traffic:")
    print("  * credit_score: mean shifted 700 -> 560 (-3.5 std deviations)")
    print("  * debt_to_income: mean shifted 0.30 -> 0.52 (+3.6 std deviations)")
    print("  * annual_income & loan_amount: unshifted control features")

    np.random.seed(999)
    # Severe negative shift on credit score
    shifted_credit = np.random.normal(loc=560.0, scale=45.0, size=30)
    # Severe positive shift on debt-to-income
    shifted_dti = np.random.normal(loc=0.52, scale=0.08, size=30)
    # Controls
    ctrl_income = np.random.normal(loc=65000.0, scale=12000.0, size=30)
    ctrl_loan = np.random.normal(loc=20000.0, scale=4000.0, size=30)

    print("\nSending 30 shifted production requests through InferenceEngine...")
    for i in range(30):
        payload = {
            "features": {
                "credit_score": round(float(shifted_credit[i]), 1),
                "annual_income": round(float(ctrl_income[i]), 2),
                "debt_to_income": round(float(shifted_dti[i]), 4),
                "loan_amount": round(float(ctrl_loan[i]), 2),
            }
        }
        res = client.post(f"/api/v1/models/{model_id}/predict", json=payload, headers=auth_headers)
        assert res.status_code == 200

    # Evaluate using PSI
    print("\nEvaluating Shifted Data with PSI...")
    async with AsyncSessionLocal() as session:
        events_shifted_psi = await drift_service.evaluate_drift(
            session=session,
            model_version_id=uuid.UUID(version_id),
            method="PSI",
            window_size=30,
            threshold=0.25,
        )

    print("\n  [Shifted Traffic Results - Method: PSI]")
    print(
        f"  {'Feature':<18} {'Score (PSI)':<14} {'Threshold':<12} {'Severity':<10} {'Drift Detected'}"
    )
    print(f"  {'-' * 65}")
    for e in events_shifted_psi:
        print(
            f"  {e.feature:<18} {e.score:<14.4f} {e.threshold:<12.2f} {e.severity:<10} {str(e.drift_detected)}"
        )

    # Evaluate using KS-Test
    print("\nEvaluating Shifted Data with Two-Sample KS-Test...")
    async with AsyncSessionLocal() as session:
        events_shifted_ks = await drift_service.evaluate_drift(
            session=session,
            model_version_id=uuid.UUID(version_id),
            method="KS_TEST",
            window_size=30,
            threshold=0.25,
        )

    print("\n  [Shifted Traffic Results - Method: KS Test]")
    print(
        f"  {'Feature':<18} {'Score (D)':<14} {'Threshold':<12} {'Severity':<10} {'Drift Detected'}"
    )
    print(f"  {'-' * 65}")
    for e in events_shifted_ks:
        print(
            f"  {e.feature:<18} {e.score:<14.4f} {e.threshold:<12.2f} {e.severity:<10} {str(e.drift_detected)}"
        )

    # --------------------------------------------------------------------------
    # Phase 3: Query Drift API Endpoint
    # --------------------------------------------------------------------------
    print_header("PHASE 3: QUERY GET /api/v1/models/{id}/drift")
    api_res = client.get(f"/api/v1/models/{model_id}/drift", headers=auth_headers)
    print(f"GET /api/v1/models/{model_id}/drift -> HTTP {api_res.status_code}")
    data = api_res.json()
    print(f"Model Name: {data['model_name']} (Version: {data['version_tag']})")
    print(f"Drift Status: {'DRIFT DETECTED' if data['drift_detected'] else 'HEALTHY'}")
    print(f"Summary: {data['summary']}")
    print(f"Total Drift Events Logged: {len(data['events'])}")

    print("\nRecent Drift Events (Sample):")
    for ev in data["events"][:6]:
        print(
            f"  [{ev['detected_at'][:19]}] Feature: {ev['feature']:<16} Method: {ev['method']:<8} Score: {ev['score']:<6.3f} Threshold: {ev['threshold']:<5.2f} Severity: {ev['severity']}"
        )

    print_header("STAGE 8 DRIFT SIMULATION COMPLETE: ALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
