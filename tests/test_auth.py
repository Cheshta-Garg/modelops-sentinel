import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints.auth import auth_rate_limiter
from app.models.monitoring import AuditLog


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Ensure rate limiter bucket is clean before each test."""
    auth_rate_limiter.reset()
    yield
    auth_rate_limiter.reset()



@pytest.mark.asyncio
async def test_register_user_success(
    unauthenticated_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Verify POST /api/v1/auth/register registers account with bcrypt hash and audit log."""
    payload = {
        "email": "engineer.test@modelops.com",
        "password": "StrongPassword123!",
        "full_name": "Test ML Engineer",
        "role": "engineer",
    }
    response = await unauthenticated_client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "engineer.test@modelops.com"
    assert data["full_name"] == "Test ML Engineer"
    assert data["role"] == "engineer"
    assert "password" not in data
    assert "hashed_password" not in data

    # Verify audit log was recorded
    audit_stmt = select(AuditLog).where(AuditLog.action == "USER_REGISTERED")
    audit = (await db_session.execute(audit_stmt)).scalar_one_or_none()
    assert audit is not None
    assert audit.entity_type == "USER"
    assert audit.user_id is not None


@pytest.mark.asyncio
async def test_register_duplicate_email_conflict(unauthenticated_client: AsyncClient) -> None:
    """Verify duplicate email registration returns 409 Conflict."""
    payload = {
        "email": "duplicate@modelops.com",
        "password": "StrongPassword123!",
        "full_name": "First User",
        "role": "viewer",
    }
    resp1 = await unauthenticated_client.post("/api/v1/auth/register", json=payload)
    assert resp1.status_code == 201

    resp2 = await unauthenticated_client.post("/api/v1/auth/register", json=payload)
    assert resp2.status_code == 409
    assert resp2.json()["error"]["code"] == "USER_ALREADY_EXISTS"


@pytest.mark.asyncio
async def test_register_short_password_rejected(unauthenticated_client: AsyncClient) -> None:
    """Verify password shorter than 8 characters is rejected by schema validation (422)."""
    payload = {
        "email": "short@modelops.com",
        "password": "short",
        "full_name": "Short Password",
        "role": "viewer",
    }
    resp = await unauthenticated_client.post("/api/v1/auth/register", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_login_success(
    unauthenticated_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Verify POST /api/v1/auth/login verifies credentials and returns JWT bearer token."""
    # 1. Register user
    reg_payload = {
        "email": "login.test@modelops.com",
        "password": "Password123!",
        "full_name": "Login Tester",
        "role": "engineer",
    }
    reg_resp = await unauthenticated_client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_resp.status_code == 201

    # 2. Login
    login_payload = {
        "email": "login.test@modelops.com",
        "password": "Password123!",
    }
    login_resp = await unauthenticated_client.post("/api/v1/auth/login", json=login_payload)
    assert login_resp.status_code == 200
    token_data = login_resp.json()

    assert "access_token" in token_data and len(token_data["access_token"]) > 20
    assert token_data["token_type"] == "bearer"
    assert token_data["expires_in"] > 0
    assert token_data["user"]["email"] == "login.test@modelops.com"
    assert token_data["user"]["role"] == "engineer"

    # Verify audit log for login
    audit_stmt = select(AuditLog).where(AuditLog.action == "USER_LOGGED_IN")
    audit = (await db_session.execute(audit_stmt)).scalar_one_or_none()
    assert audit is not None


@pytest.mark.asyncio
async def test_login_invalid_password_returns_401(unauthenticated_client: AsyncClient) -> None:
    """Verify wrong password returns 401 INVALID_CREDENTIALS."""
    # Register user
    reg_payload = {
        "email": "wrongpwd@modelops.com",
        "password": "CorrectPassword123!",
        "full_name": "Wrong Pwd User",
        "role": "viewer",
    }
    await unauthenticated_client.post("/api/v1/auth/register", json=reg_payload)

    # Attempt login with wrong password
    login_resp = await unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"email": "wrongpwd@modelops.com", "password": "IncorrectPassword!"},
    )
    assert login_resp.status_code == 401
    assert login_resp.json()["error"]["code"] == "INVALID_CREDENTIALS"


@pytest.mark.asyncio
async def test_get_current_user_profile_via_token(unauthenticated_client: AsyncClient) -> None:
    """Verify GET /api/v1/auth/me resolves identity from Authorization Bearer token."""
    reg_payload = {
        "email": "profile@modelops.com",
        "password": "Password123!",
        "full_name": "Profile Tester",
        "role": "engineer",
    }
    await unauthenticated_client.post("/api/v1/auth/register", json=reg_payload)

    login_resp = await unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"email": "profile@modelops.com", "password": "Password123!"},
    )
    token = login_resp.json()["access_token"]

    # Call /auth/me with Bearer token
    me_resp = await unauthenticated_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["email"] == "profile@modelops.com"
    assert me_data["role"] == "engineer"


@pytest.mark.asyncio
async def test_protected_endpoint_without_token_returns_401(unauthenticated_client: AsyncClient) -> None:
    """Verify unauthenticated request to protected endpoint returns 401 UNAUTHENTICATED."""
    response = await unauthenticated_client.get("/api/v1/models")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


@pytest.mark.asyncio
async def test_protected_endpoint_with_invalid_token_returns_401(unauthenticated_client: AsyncClient) -> None:
    """Verify request with malformed/tampered JWT returns 401 INVALID_TOKEN."""
    response = await unauthenticated_client.get(
        "/api/v1/models",
        headers={"Authorization": "Bearer invalid.fake.token"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_TOKEN"


@pytest.mark.asyncio
async def test_rbac_viewer_forbidden_from_creating_model(unauthenticated_client: AsyncClient) -> None:
    """Verify VIEWER role gets 403 Forbidden when attempting to register a model."""
    # 1. Register VIEWER
    reg_payload = {
        "email": "viewer@modelops.com",
        "password": "Password123!",
        "full_name": "Viewer User",
        "role": "viewer",
    }
    await unauthenticated_client.post("/api/v1/auth/register", json=reg_payload)
    login_resp = await unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"email": "viewer@modelops.com", "password": "Password123!"},
    )
    viewer_token = login_resp.json()["access_token"]

    # 2. Viewer attempts POST /models
    model_payload = {
        "name": "viewer_illegal_model",
        "task_type": "binary_classification",
    }
    create_resp = await unauthenticated_client.post(
        "/api/v1/models",
        json=model_payload,
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert create_resp.status_code == 403
    assert create_resp.json()["error"]["code"] == "FORBIDDEN_INSUFFICIENT_PERMISSIONS"


@pytest.mark.asyncio
async def test_rbac_engineer_can_create_model(
    unauthenticated_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Verify ENGINEER role is authorized to create models (201 Created) and audit log is created."""
    # 1. Register ENGINEER
    reg_payload = {
        "email": "engineer.lead@modelops.com",
        "password": "Password123!",
        "full_name": "Lead ML Engineer",
        "role": "engineer",
    }
    await unauthenticated_client.post("/api/v1/auth/register", json=reg_payload)
    login_resp = await unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"email": "engineer.lead@modelops.com", "password": "Password123!"},
    )
    engineer_token = login_resp.json()["access_token"]

    # 2. Engineer creates model
    model_payload = {
        "name": "engineer_permitted_model",
        "task_type": "binary_classification",
    }
    create_resp = await unauthenticated_client.post(
        "/api/v1/models",
        json=model_payload,
        headers={"Authorization": f"Bearer {engineer_token}"},
    )
    assert create_resp.status_code == 201
    assert create_resp.json()["name"] == "engineer_permitted_model"

    # 3. Verify audit log entry MODEL_CREATED was recorded
    audit_stmt = select(AuditLog).where(
        AuditLog.action == "MODEL_CREATED",
        AuditLog.entity_type == "MODEL",
    )
    audit = (await db_session.execute(audit_stmt)).scalar_one_or_none()
    assert audit is not None


@pytest.mark.asyncio
async def test_rbac_admin_has_full_access(unauthenticated_client: AsyncClient) -> None:
    """Verify ADMIN role has full access to all endpoints."""
    reg_payload = {
        "email": "superadmin@modelops.com",
        "password": "Password123!",
        "full_name": "Super Admin",
        "role": "admin",
    }
    await unauthenticated_client.post("/api/v1/auth/register", json=reg_payload)
    login_resp = await unauthenticated_client.post(
        "/api/v1/auth/login",
        json={"email": "superadmin@modelops.com", "password": "Password123!"},
    )
    admin_token = login_resp.json()["access_token"]

    # Admin creates model
    create_resp = await unauthenticated_client.post(
        "/api/v1/models",
        json={"name": "admin_created_model", "task_type": "clf"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_resp.status_code == 201

    # Admin lists models
    list_resp = await unauthenticated_client.get(
        "/api/v1/models",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_resp.status_code == 200


@pytest.mark.asyncio
async def test_rate_limiter_blocks_excessive_auth_requests(unauthenticated_client: AsyncClient) -> None:
    """Verify rate limiter triggers 429 Too Many Requests when requests exceed threshold."""
    # Attempt rapid repeated logins from the same client
    payload = {"email": "ratelimit@modelops.com", "password": "WrongPassword!"}
    status_codes = []
    for _ in range(15):
        resp = await unauthenticated_client.post("/api/v1/auth/login", json=payload)
        status_codes.append(resp.status_code)

    assert 429 in status_codes
    rate_limited_index = status_codes.index(429)
    assert rate_limited_index <= 11  # Rate limit is set to 10
