import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_liveness_probe_returns_200(async_client: AsyncClient) -> None:
    """Verify /health returns 200 OK and valid service metadata."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ModelOps Sentinel" in data["service"]
    assert "version" in data
    assert "environment" in data


@pytest.mark.asyncio
async def test_request_id_middleware_generates_header(async_client: AsyncClient) -> None:
    """Verify middleware injects X-Request-ID and X-Response-Time-Ms if omitted."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers
    assert "x-response-time-ms" in response.headers
    assert len(response.headers["x-request-id"]) > 10


@pytest.mark.asyncio
async def test_request_id_middleware_preserves_client_header(
    async_client: AsyncClient,
) -> None:
    """Verify middleware propagates an existing client X-Request-ID header."""
    custom_id = "trace-uuid-abcdef-123456"
    response = await async_client.get(
        "/api/v1/health",
        headers={"X-Request-ID": custom_id},
    )
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id


@pytest.mark.asyncio
async def test_readiness_probe_disconnected_subsystems(async_client: AsyncClient) -> None:
    """Verify /ready returns 503 when backend services (Postgres, Redis) are offline."""
    response = await async_client.get("/api/v1/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert "checks" in data
    assert data["checks"]["postgres"] == "disconnected"
    assert data["checks"]["redis"] == "disconnected"


@pytest.mark.asyncio
async def test_readiness_probe_connected_subsystems(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /ready returns 200 OK when backend subsystems are healthy."""
    from app.api.v1.endpoints import health

    async def mock_pg() -> dict[str, str]:
        return {"status": "connected"}

    async def mock_redis() -> dict[str, str]:
        return {"status": "connected"}

    monkeypatch.setattr(health, "check_postgres", mock_pg)
    monkeypatch.setattr(health, "check_redis", mock_redis)

    response = await async_client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["checks"]["postgres"] == "connected"
    assert data["checks"]["redis"] == "connected"


@pytest.mark.asyncio
async def test_handled_app_exception_flow(async_client: AsyncClient) -> None:
    """Verify AppException returns structured error format and correct status."""
    response = await async_client.get("/api/v1/test-error?error_type=app")
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "DIAGNOSTIC_TEST_ERROR"
    assert "This is a controlled diagnostic application error." in data["error"]["message"]
    assert data["error"]["request_id"] == response.headers.get("x-request-id")


@pytest.mark.asyncio
async def test_unhandled_exception_flow(async_client: AsyncClient) -> None:
    """Verify unhandled 500 error returns safe standardized response without leaking stack trace."""
    response = await async_client.get("/api/v1/test-error?error_type=unexpected")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert data["error"]["request_id"] == response.headers.get("x-request-id")
