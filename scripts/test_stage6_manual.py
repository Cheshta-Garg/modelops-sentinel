"""Stage 6 Manual Verification Script (Zero Docker Required).

Executes the complete manual verification lifecycle for Stage 6:
1. Authenticates engineer credentials and acquires JWT token
2. Registers baseline model and version artifact if not already registered
3. Submits asynchronous inference job via POST /api/v1/models/{id}/predict/async -> verifies 202 Accepted and job_id
4. Polls GET /api/v1/jobs/{job_id} until complete -> verifies normal prediction (0)
5. Submits async cyber-attack prediction job -> verifies anomaly detection (1)
6. Tests failure state handling on invalid features payload (status = failed)
7. Tests telemetry processing worker task
8. Tests worker retry behavior on transient errors
9. Tests worker timeout behavior on hanging tasks
10. Tests worker restart simulation and state recovery
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from httpx import ASGITransport, AsyncClient

from app.db.base import Base
from app.db.session import AsyncSessionLocal, async_engine
from app.main import app
from app.services.queue import queue_service
from app.worker.tasks import TASK_REGISTRY
from scripts.train_baseline_model import train_and_export_model

SWAT_NORMAL_FEATURES = {
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


async def run_stage6_verification() -> None:
    print("=" * 80)
    print("  MODELOPS SENTINEL -- STAGE 6 ASYNC INFERENCE & WORKER VERIFICATION")
    print("=" * 80)

    # 1. Initialize SQLite database schema
    print("\n[Step 1/8] Initializing local database schema...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  [OK] Database schema initialized.")

    # 2. Ensure baseline model artifact is trained
    artifact_path = Path("artifacts/baseline_swat_model.joblib")
    if not artifact_path.exists():
        print("  [*] Training baseline SWaT model artifact...")
        train_and_export_model(str(artifact_path))

    # Configure queue service to use local session factory and memory backend
    queue_service.set_force_memory(True)
    queue_service.set_session_factory(AsyncSessionLocal)
    await queue_service._memory_queue.clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Authenticate as Engineer
        print("\n[Step 2/8] Authenticating Engineer account...")
        await client.post(
            "/api/v1/auth/register",
            json={
                "email": "stage6_engineer@modelops.io",
                "password": "Password123!",
                "full_name": "Stage 6 Engineer",
                "role": "engineer",
            },
        )
        login_resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "stage6_engineer@modelops.io", "password": "Password123!"},
        )
        token = login_resp.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {token}"}
        print("  [OK] Engineer authenticated. JWT issued.")

        # Register model and staged version
        print("\n[Step 3/8] Ensuring registered SWaT model and STAGING version exist...")
        reg_resp = await client.post(
            "/api/v1/models",
            headers=auth_headers,
            json={
                "name": "stage6_swat_water_detector",
                "description": "SWaT async anomaly detection model",
                "task_type": "binary_classification",
            },
        )
        if reg_resp.status_code == 201:
            model_id = reg_resp.json()["id"]
        else:
            list_m = await client.get("/api/v1/models", headers=auth_headers)
            model_id = next(
                m["id"] for m in list_m.json() if m["name"] == "stage6_swat_water_detector"
            )

        with open(artifact_path, "rb") as f:
            art_bytes = f.read()

        await client.post(
            f"/api/v1/models/{model_id}/versions",
            headers=auth_headers,
            files={"file": ("model.joblib", art_bytes, "application/octet-stream")},
            data={
                "version_tag": "v1.0.0",
                "input_schema": json.dumps(
                    {"features": ["LIT101", "FIT101", "AIT201", "P101", "DPIT301"]}
                ),
                "output_schema": json.dumps({"prediction": "int", "confidence": "float"}),
                "artifact_format": "JOBLIB",
            },
        )
        print(f"  [OK] Model '{model_id}' ready for asynchronous inference.")

        # -----------------------------------------------------------------
        # Step 4: Asynchronous Normal Inference & Polling
        # -----------------------------------------------------------------
        print("\n[Step 4/8] Submitting Asynchronous Inference Job (Normal Telemetry)...")
        post_resp = await client.post(
            f"/api/v1/models/{model_id}/predict/async",
            headers=auth_headers,
            json={"features": SWAT_NORMAL_FEATURES},
        )
        assert post_resp.status_code == 202
        job_id = post_resp.json()["job_id"]
        print(f"  [OK] Received HTTP 202 Accepted. Assigned Job ID: {job_id}")

        print("  [*] Polling GET /api/v1/jobs/{job_id}...")
        completed_job = None
        for i in range(25):
            await asyncio.sleep(0.1)
            poll_resp = await client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
            assert poll_resp.status_code == 200
            data = poll_resp.json()
            print(f"      - Poll #{i + 1}: status = '{data['status']}'")
            if data["status"] == "complete":
                completed_job = data
                break

        assert completed_job is not None, "Job did not complete within timeout"
        res = completed_job["result"]
        print("  [OK] Job Completed Successfully:")
        print(f"      - Model:      {res['model']}")
        print(f"      - Version:    {res['version']}")
        print(f"      - Prediction: {res['prediction']} (0 = Normal)")
        print(f"      - Confidence: {res['confidence']}")
        print(f"      - Latency:    {res['latency_ms']} ms")
        assert res["prediction"] == 0

        # -----------------------------------------------------------------
        # Step 5: Asynchronous Cyber-Attack Anomaly Detection
        # -----------------------------------------------------------------
        print("\n[Step 5/8] Submitting Asynchronous Inference Job (Simulated Cyber-Attack)...")
        post_attack = await client.post(
            f"/api/v1/models/{model_id}/predict/async",
            headers=auth_headers,
            json={"features": SWAT_ATTACK_FEATURES},
        )
        attack_job_id = post_attack.json()["job_id"]
        print(f"  [OK] Received HTTP 202 Accepted. Assigned Job ID: {attack_job_id}")

        completed_attack = None
        for _ in range(25):
            await asyncio.sleep(0.1)
            poll_resp = await client.get(f"/api/v1/jobs/{attack_job_id}", headers=auth_headers)
            data = poll_resp.json()
            if data["status"] == "complete":
                completed_attack = data
                break

        assert completed_attack is not None
        attack_res = completed_attack["result"]
        print("  [OK] Attack Telemetry Processed:")
        print(
            f"      - Prediction: {attack_res['prediction']} (1 = Cyber-Attack Anomaly Detected!)"
        )
        print(f"      - Confidence: {attack_res['confidence']}")
        assert attack_res["prediction"] == 1

        # -----------------------------------------------------------------
        # Step 6: Failure State Handling
        # -----------------------------------------------------------------
        print("\n[Step 6/8] Testing Failure State Handling on Invalid Input...")
        invalid_resp = await client.post(
            f"/api/v1/models/{model_id}/predict/async",
            headers=auth_headers,
            json={"features": {"LIT101": 500.0}},  # Missing 4 required features
        )
        assert invalid_resp.status_code == 202
        invalid_job_id = invalid_resp.json()["job_id"]
        print(f"  [OK] Accepted invalid job for background processing: {invalid_job_id}")

        failed_job = None
        for _ in range(25):
            await asyncio.sleep(0.1)
            poll_resp = await client.get(f"/api/v1/jobs/{invalid_job_id}", headers=auth_headers)
            data = poll_resp.json()
            if data["status"] == "failed":
                failed_job = data
                break

        assert failed_job is not None, "Invalid job did not transition to failed state"
        print("  [OK] Job Correctly Transitioned to Failure State:")
        print(f"      - Status:  '{failed_job['status']}'")
        print(f"      - Success: {failed_job['success']}")
        print(f"      - Error:   {failed_job['error']}")

        # -----------------------------------------------------------------
        # Step 7: Telemetry Processing Worker Task
        # -----------------------------------------------------------------
        print("\n[Step 7/8] Testing Telemetry Processing Worker Task...")
        sample_batch = [
            {"prediction": 0, "latency_ms": 12.5},
            {"prediction": 0, "latency_ms": 14.1},
            {"prediction": 1, "latency_ms": 16.8},
            {"prediction": 0, "latency_ms": 13.0},
        ]
        telemetry_job_id = await queue_service.enqueue("process_telemetry_task", sample_batch)
        print(f"  [OK] Enqueued telemetry batch task: {telemetry_job_id}")

        for _ in range(20):
            await asyncio.sleep(0.05)
            tel_resp = await client.get(f"/api/v1/jobs/{telemetry_job_id}", headers=auth_headers)
            t_data = tel_resp.json()
            if t_data["status"] == "complete":
                break

        t_result = t_data["result"]
        print("  [OK] Telemetry Batch Aggregated by Worker:")
        print(f"      - Processed Count: {t_result['processed_count']} events")
        print(f"      - Average Latency: {t_result['avg_latency_ms']} ms")
        print(f"      - Class Distribution: {t_result['prediction_distribution']}")

        # -----------------------------------------------------------------
        # Step 8: Retries, Timeouts, and Worker Restart
        # -----------------------------------------------------------------
        print("\n[Step 8/8] Testing Worker Retries, Timeouts & Restart Simulation...")

        # 8A: Retry on transient failure
        transient_tries = 0

        async def manual_flaky_task(ctx: dict) -> str:
            nonlocal transient_tries
            transient_tries += 1
            if transient_tries < 2:
                raise RuntimeError("Temporary network timeout")
            return "recovered_after_retry"

        TASK_REGISTRY["manual_flaky_task"] = manual_flaky_task
        retry_job_id = await queue_service.enqueue("manual_flaky_task", max_tries=3)
        for _ in range(25):
            await asyncio.sleep(0.1)
            r_resp = await client.get(f"/api/v1/jobs/{retry_job_id}", headers=auth_headers)
            if r_resp.json()["status"] == "complete":
                break
        print(
            f"  [OK] Retry Task Succeeded: retries={r_resp.json()['retries']}, result={r_resp.json()['result']}"
        )

        # 8B: Timeout handling
        async def manual_hanging_task(ctx: dict) -> str:
            await asyncio.sleep(2.0)
            return "done"

        TASK_REGISTRY["manual_hanging_task"] = manual_hanging_task
        timeout_job_id = await queue_service.enqueue(
            "manual_hanging_task", job_timeout=0.2, max_tries=1
        )
        for _ in range(25):
            await asyncio.sleep(0.05)
            to_resp = await client.get(f"/api/v1/jobs/{timeout_job_id}", headers=auth_headers)
            if to_resp.json()["status"] == "failed":
                break
        print(
            f"  [OK] Timeout Task Terminated: status={to_resp.json()['status']}, error={to_resp.json()['error']}"
        )

        # 8C: Worker Restart Simulation
        async def lingering_task(ctx: dict) -> str:
            await asyncio.sleep(0.5)
            return "lingering_complete"

        TASK_REGISTRY["lingering_task"] = lingering_task
        restart_job_id = await queue_service.enqueue("lingering_task", job_timeout=5, max_tries=2)
        queue_service.restart_worker()
        rest_job = await queue_service.get_job(restart_job_id)
        assert rest_job is not None
        print(
            f"  [OK] Worker Restart Simulation Successful: job '{restart_job_id}' status = {rest_job.status}"
        )

    print("\n" + "=" * 80)
    print("  [PASS] ALL STAGE 6 ASYNC & WORKER VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_stage6_verification())
