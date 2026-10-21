"""Stage 4 Manual Testing Script (Zero Docker Required).

Executes the complete manual testing lifecycle for Stage 4:
1. Auto-trains baseline SWaT anomaly model
2. Registers logical model identity in catalog
3. Uploads model artifact version (runs load-test -> STAGING)
4. Executes normal inference (Cache Miss)
5. Executes identical repeat inference (Cache Hit)
6. Executes simulated cyber-attack inference (Anomaly Detection)
7. Tests error handling on invalid schema (422)
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from httpx import ASGITransport, AsyncClient

from app.db.base import Base
from app.db.session import async_engine
from app.main import app
from scripts.train_baseline_model import train_and_export_model


async def run_manual_test() -> None:
    print("=" * 70)
    print("MODELOPS SENTINEL -- STAGE 4 MANUAL VERIFICATION (ZERO DOCKER)")
    print("=" * 70)

    # 1. Initialize SQLite database tables
    print("\n[Step 1/5] Initializing local database schema...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  [OK] Database schema ready (SQLite)")

    # 2. Ensure baseline model artifact is trained
    artifact_path = Path("artifacts/baseline_swat_model.joblib")
    if not artifact_path.exists():
        print("\n[Step 2/5] Training baseline SWaT anomaly model...")
        train_and_export_model(str(artifact_path))
    else:
        print(f"\n[Step 2/5] Found existing baseline model: {artifact_path}")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Authenticate as Engineer
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "stage4_engineer@modelops.io",
                "password": "Password123!",
                "full_name": "Stage 4 Engineer",
                "role": "engineer",
            },
        )
        login_resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "stage4_engineer@modelops.io", "password": "Password123!"},
        )
        token = login_resp.json()["access_token"]

        auth_headers = {"Authorization": f"Bearer {token}"}

        # 3. Register logical model
        print("\n[Step 3/5] Registering logical model in catalog...")
        reg_resp = await client.post(
            "/api/v1/models",
            headers=auth_headers,
            json={
                "name": "swat_water_anomaly_detector",
                "description": "Real-time water treatment sensor anomaly detector",
                "task_type": "binary_classification",
            },
        )
        if reg_resp.status_code == 201:
            model_id = reg_resp.json()["id"]
            print(f"  [OK] Model registered: swat_water_anomaly_detector (id: {model_id})")
        elif reg_resp.status_code == 409:
            # Already exists from previous run, fetch it
            list_resp = await client.get("/api/v1/models", headers=auth_headers)
            model = next(m for m in list_resp.json() if m["name"] == "swat_water_anomaly_detector")
            model_id = model["id"]
            print(f"  [INFO] Model already exists: swat_water_anomaly_detector (id: {model_id})")
        else:
            raise RuntimeError(f"Failed to register model: {reg_resp.text}")

        # 4. Upload model version artifact
        print("\n[Step 4/5] Uploading model version artifact...")
        with open(artifact_path, "rb") as f:
            artifact_bytes = f.read()

        ver_resp = await client.post(
            f"/api/v1/models/{model_id}/versions",
            headers=auth_headers,
            files={"file": ("model.joblib", artifact_bytes, "application/octet-stream")},
            data={
                "version_tag": "v1.0.0",
                "input_schema": json.dumps({"features": ["LIT101", "FIT101", "AIT201", "P101", "DPIT301"]}),
                "output_schema": json.dumps({"prediction": "int", "confidence": "float"}),
                "artifact_format": "JOBLIB",
            },
        )
        if ver_resp.status_code == 201:
            v_data = ver_resp.json()
            print(f"  [OK] Version uploaded and staged: {v_data['version_tag']} (status: {v_data['status']})")
        elif ver_resp.status_code == 409:
            print("  [INFO] Version v1.0.0 already uploaded and staged.")
        else:
            raise RuntimeError(f"Failed to upload version: {ver_resp.text}")

        # 5. Run inference tests
        print("\n[Step 5/5] Executing Synchronous Inference Tests...")

        # Test 5A: Normal conditions (Cache Miss)
        print("\n  [*] Test 5A: Normal Process Telemetry (Expected: Cache Miss)")
        normal_payload = {
            "features": {
                "LIT101": 650.0,
                "FIT101": 2.45,
                "AIT201": 255.0,
                "P101": 1,
                "DPIT301": 19.5,
            }
        }
        r1 = await client.post(f"/api/v1/models/{model_id}/predict", headers=auth_headers, json=normal_payload)
        d1 = r1.json()
        print(f"      - Status:     {r1.status_code} OK")
        print(f"      - Request ID: {d1['request_id']}")
        print(f"      - Prediction: {d1['prediction']} (0 = Normal)")
        print(f"      - Confidence: {d1['confidence']}")
        print(f"      - Latency:    {d1['latency_ms']} ms")
        print(f"      - Cached:     {d1['cached']} (Cache Miss)")
        assert d1["prediction"] == 0
        assert d1["cached"] is False

        # Test 5B: Identical repeat query (Cache Hit)
        print("\n  [*] Test 5B: Repeat Query with Identical Features (Expected: Cache Hit)")
        r2 = await client.post(f"/api/v1/models/{model_id}/predict", headers=auth_headers, json=normal_payload)
        d2 = r2.json()
        print(f"      - Status:     {r2.status_code} OK")
        print(f"      - Prediction: {d2['prediction']} (0 = Normal)")
        print(f"      - Confidence: {d2['confidence']}")
        print(f"      - Latency:    {d2['latency_ms']} ms")
        print(f"      - Cached:     {d2['cached']} (Cache Hit!)")
        assert d2["prediction"] == 0
        assert d2["cached"] is True

        # Test 5C: Simulated Cyber-Attack (Anomaly Detected)
        print("\n  [*] Test 5C: Simulated Cyber-Attack Anomaly (Expected: Prediction = 1)")
        attack_payload = {
            "features": {
                "LIT101": 950.0,   # Tank overflow surge
                "FIT101": 5.80,    # High inflow rate
                "AIT201": 380.0,   # Chemical tampering
                "P101": 0,
                "DPIT301": 48.0,   # Dangerous filter pressure surge
            }
        }
        r3 = await client.post(f"/api/v1/models/{model_id}/predict", headers=auth_headers, json=attack_payload)
        d3 = r3.json()
        print(f"      - Status:     {r3.status_code} OK")
        print(f"      - Prediction: {d3['prediction']} (1 = Cyber-Attack Anomaly Detected!)")
        print(f"      - Confidence: {d3['confidence']}")
        print(f"      - Latency:    {d3['latency_ms']} ms")
        print(f"      - Cached:     {d3['cached']}")
        assert d3["prediction"] == 1

        # Test 5D: Invalid Input Schema (Missing features)
        print("\n  [*] Test 5D: Input Validation Error (Expected: 422 Unprocessable Entity)")
        invalid_payload = {"features": {"LIT101": 650.0}}
        r4 = await client.post(f"/api/v1/models/{model_id}/predict", headers=auth_headers, json=invalid_payload)
        d4 = r4.json()
        print(f"      - Status:     {r4.status_code} Unprocessable Entity")
        print(f"      - Error Code: {d4['error']['code']}")
        print(f"      - Message:    {d4['error']['message']}")
        assert r4.status_code == 422
        assert d4["error"]["code"] == "INVALID_INPUT_FEATURES"

    print("\n" + "=" * 70)
    print("[PASS] ALL STAGE 4 MANUAL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_manual_test())
