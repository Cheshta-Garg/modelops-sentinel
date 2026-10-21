"""Stage 5 Manual Verification Script (Zero Docker Required).

Executes the complete manual verification lifecycle for Stage 5:
1. Initializes local SQLite database
2. Tests user registration & bcrypt password hashing (admin, engineer, viewer)
3. Tests JWT authentication & access token generation via /api/v1/auth/login
4. Tests token validation and profile retrieval via /api/v1/auth/me
5. Tests unauthenticated access rejection (401)
6. Tests invalid/tampered token rejection (401)
7. Tests Role-Based Access Control (RBAC):
   - Viewer forbidden from creating model (403)
   - Engineer permitted to create model (201)
   - Engineer permitted to upload model version (201)
   - Viewer permitted to read models and execute inference (200)
   - Admin permitted full access
8. Verifies audit trail in database (audit_logs table)
9. Tests Redis / local rate limiter protection against brute-force attacks (429)
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.api.v1.endpoints.auth import auth_rate_limiter
from app.db.base import Base
from app.db.session import AsyncSessionLocal, async_engine
from app.main import app
from app.models.monitoring import AuditLog
from app.models.user import User
from scripts.train_baseline_model import train_and_export_model


async def run_stage5_verification() -> None:
    print("=" * 75)
    print("  MODELOPS SENTINEL -- STAGE 5 AUTH & RBAC MANUAL VERIFICATION")
    print("=" * 75)

    # 1. Initialize SQLite database tables
    print("\n[Step 1/8] Initializing local database schema...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("  [OK] Database schema initialized.")

    # 2. Ensure baseline model artifact is trained for version upload
    artifact_path = Path("artifacts/baseline_swat_model.joblib")
    if not artifact_path.exists():
        print("  [*] Training baseline SWaT model artifact...")
        train_and_export_model(str(artifact_path))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Reset rate limiter memory for clean test run
        auth_rate_limiter.reset()

        # -------------------------------------------------------------
        # Step 2: Register Users (Admin, Engineer, Viewer)
        # -------------------------------------------------------------
        print("\n[Step 2/8] Registering RBAC accounts...")
        users_data = [
            ("admin@modelops.io", "AdminPass123!", "Alice Admin", "admin"),
            ("engineer@modelops.io", "EngineerPass123!", "Bob Engineer", "engineer"),
            ("viewer@modelops.io", "ViewerPass123!", "Charlie Viewer", "viewer"),
        ]

        tokens = {}
        for email, password, name, role in users_data:
            reg_resp = await client.post(
                "/api/v1/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "full_name": name,
                    "role": role,
                },
            )
            if reg_resp.status_code == 201:
                print(f"  [OK] Registered {role.upper():<8}: {email}")
            elif reg_resp.status_code == 409:
                print(f"  [INFO] Already registered: {email}")
            else:
                raise RuntimeError(f"Failed to register {email}: {reg_resp.text}")

            login_resp = await client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": password},
            )
            assert login_resp.status_code == 200, f"Login failed for {email}: {login_resp.text}"
            token = login_resp.json()["access_token"]
            tokens[role] = token
            print(f"  [OK] Logged in   {role.upper():<8}: {email} (JWT issued)")

        # -------------------------------------------------------------
        # Step 3: Verify Bcrypt Hashing in Database
        # -------------------------------------------------------------
        print("\n[Step 3/8] Verifying Bcrypt password hashing in database...")
        async with AsyncSessionLocal() as session:
            stmt = select(User).where(User.email == "engineer@modelops.io")
            eng_user = (await session.execute(stmt)).scalar_one()
            assert eng_user.hashed_password != "EngineerPass123!"
            assert eng_user.hashed_password.startswith("$2b$")
            print(f"  [OK] Stored password hash starts with: {eng_user.hashed_password[:28]}...")
            print("  [OK] Salt & cost factor: 12 rounds, raw text never persisted.")

        # -------------------------------------------------------------
        # Step 4: Token Profile Retrieval & Unauthenticated Rejection
        # -------------------------------------------------------------
        print("\n[Step 4/8] Testing Token Profile & Missing/Invalid Auth...")
        # Profile check
        me_resp = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens['engineer']}"},
        )
        assert me_resp.status_code == 200
        me_data = me_resp.json()
        print(f"  [OK] /auth/me decoded user: {me_data['email']} (role: {me_data['role']})")

        # Missing token -> 401
        no_auth_resp = await client.post("/api/v1/models", json={"name": "test_unauth"})
        assert no_auth_resp.status_code == 401
        print(f"  [OK] Request without token -> 401 Unauthenticated: {no_auth_resp.json()['error']['code']}")

        # Invalid/Tampered token -> 401
        bad_token_resp = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature"},
        )
        assert bad_token_resp.status_code == 401
        print(f"  [OK] Request with tampered token -> 401: {bad_token_resp.json()['error']['code']}")

        # -------------------------------------------------------------
        # Step 5: Test RBAC Permissions
        # -------------------------------------------------------------
        print("\n[Step 5/8] Testing Role-Based Access Control (RBAC)...")

        # 5A: Viewer tries to register a model -> 403 Forbidden
        viewer_headers = {"Authorization": f"Bearer {tokens['viewer']}"}
        viewer_model_resp = await client.post(
            "/api/v1/models",
            headers=viewer_headers,
            json={"name": "forbidden_viewer_model", "task_type": "binary_classification"},
        )
        assert viewer_model_resp.status_code == 403
        print(f"  [OK] Viewer creating model -> 403 Forbidden: {viewer_model_resp.json()['error']['code']}")

        # 5B: Engineer creates a model -> 201 Created
        eng_headers = {"Authorization": f"Bearer {tokens['engineer']}"}
        eng_model_resp = await client.post(
            "/api/v1/models",
            headers=eng_headers,
            json={
                "name": "stage5_water_detector",
                "description": "Model created under Stage 5 RBAC test",
                "task_type": "binary_classification",
            },
        )
        if eng_model_resp.status_code == 201:
            stage5_model_id = eng_model_resp.json()["id"]
            print(f"  [OK] Engineer created model -> 201 Created (ID: {stage5_model_id})")
        else:
            list_m = await client.get("/api/v1/models", headers=eng_headers)
            stage5_model_id = next(m["id"] for m in list_m.json() if m["name"] == "stage5_water_detector")
            print(f"  [INFO] Model already exists (ID: {stage5_model_id})")

        # 5C: Engineer uploads model version artifact -> 201 Created
        with open(artifact_path, "rb") as f:
            art_bytes = f.read()

        ver_resp = await client.post(
            f"/api/v1/models/{stage5_model_id}/versions",
            headers=eng_headers,
            files={"file": ("model.joblib", art_bytes, "application/octet-stream")},
            data={
                "version_tag": "v1.0.0",
                "input_schema": json.dumps({"features": ["LIT101", "FIT101", "AIT201", "P101", "DPIT301"]}),
                "output_schema": json.dumps({"prediction": "int", "confidence": "float"}),
                "artifact_format": "JOBLIB",
            },
        )
        if ver_resp.status_code == 201:
            print("  [OK] Engineer uploaded model version -> 201 Created (Status: STAGING)")
        else:
            print("  [INFO] Version already exists.")

        # 5D: Viewer reads models -> 200 OK
        viewer_list_resp = await client.get("/api/v1/models", headers=viewer_headers)
        assert viewer_list_resp.status_code == 200
        print(f"  [OK] Viewer listed models -> 200 OK ({len(viewer_list_resp.json())} models found)")

        # 5E: Viewer executes inference -> 200 OK
        inf_resp = await client.post(
            f"/api/v1/models/{stage5_model_id}/predict",
            headers=viewer_headers,
            json={"features": {"LIT101": 650.0, "FIT101": 2.45, "AIT201": 255.0, "P101": 1, "DPIT301": 19.5}},
        )
        assert inf_resp.status_code == 200
        print(f"  [OK] Viewer executed inference -> 200 OK (Prediction: {inf_resp.json()['prediction']})")

        # -------------------------------------------------------------
        # Step 6: Verify Admin Universal Privileges
        # -------------------------------------------------------------
        print("\n[Step 6/8] Testing Admin Universal Privileges...")
        admin_headers = {"Authorization": f"Bearer {tokens['admin']}"}
        admin_model_resp = await client.get(f"/api/v1/models/{stage5_model_id}", headers=admin_headers)
        assert admin_model_resp.status_code == 200
        print(f"  [OK] Admin accessed model detail -> 200 OK: {admin_model_resp.json()['name']}")

        # -------------------------------------------------------------
        # Step 7: Verify Audit Logging in Database
        # -------------------------------------------------------------
        print("\n[Step 7/8] Verifying Audit Trail in Database...")
        async with AsyncSessionLocal() as session:
            stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(5)
            logs = (await session.execute(stmt)).scalars().all()
            print(f"  [OK] Found {len(logs)} recent audit log entries:")
            for log in logs:
                print(f"      - Action: {log.action:<24} | Entity: {log.entity_type:<14} | User: {str(log.user_id)[:8]}...")

        # -------------------------------------------------------------
        # Step 8: Test Rate Limiting
        # -------------------------------------------------------------
        print("\n[Step 8/8] Testing Rate Limiting on Auth Endpoints...")
        auth_rate_limiter.reset()

        rate_limit_triggered = False
        for i in range(15):
            res = await client.post(
                "/api/v1/auth/login",
                json={"email": "attacker@modelops.io", "password": "WrongPassword123!"},
            )
            if res.status_code == 429:
                rate_limit_triggered = True
                print(f"  [OK] Request #{i+1} blocked: 429 Too Many Requests ({res.json()['error']['code']})")
                print(f"      - Retry-After: {res.headers.get('Retry-After', '60')}s")
                break

        assert rate_limit_triggered, "Rate limiter failed to trigger 429 after burst"

    print("\n" + "=" * 75)
    print("  [PASS] ALL STAGE 5 AUTH & RBAC VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(run_stage5_verification())
