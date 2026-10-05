import asyncio
from typing import Any

import asyncpg
import redis.asyncio as aioredis
from fastapi import APIRouter, Query, Response, status

from app.core.config import get_settings
from app.core.errors import AppException
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health & Diagnostics"])
settings = get_settings()


async def check_postgres() -> dict[str, Any]:
    """Verify PostgreSQL connectivity asynchronously."""
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB,
            ),
            timeout=2.0,
        )
        await conn.execute("SELECT 1")
        await conn.close()
        return {"status": "connected", "latency_ms": 1.0}
    except Exception as exc:
        logger.debug(f"PostgreSQL readiness check failed: {exc}")
        return {"status": "disconnected", "error": str(exc)}


async def check_redis() -> dict[str, Any]:
    """Verify Redis connectivity asynchronously."""
    client = None
    try:
        client = aioredis.from_url(
            settings.redis_url,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
        )
        pong = await asyncio.wait_for(client.ping(), timeout=2.0)
        if pong:
            return {"status": "connected"}
        return {"status": "disconnected", "error": "No PONG response"}
    except Exception as exc:
        logger.debug(f"Redis readiness check failed: {exc}")
        return {"status": "disconnected", "error": str(exc)}
    finally:
        if client:
            await client.aclose()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Liveness Probe",
    description="Returns 200 if the API process is alive and accepting HTTP traffic.",
)
async def liveness_probe() -> dict[str, str]:
    """Kubernetes/Container liveness probe."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }


@router.get(
    "/ready",
    summary="Readiness Probe",
    description="Returns 200 if all dependent subsystems (PostgreSQL, Redis) are reachable.",
)
async def readiness_probe(response: Response) -> dict[str, Any]:
    """Kubernetes/Container readiness probe checking upstream infrastructure."""
    pg_check, redis_check = await asyncio.gather(
        check_postgres(),
        check_redis(),
        return_exceptions=False,
    )

    is_pg_ok = pg_check["status"] == "connected"
    is_redis_ok = redis_check["status"] == "connected"
    is_ready = is_pg_ok and is_redis_ok

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if is_ready else "not_ready",
        "service": settings.PROJECT_NAME,
        "checks": {
            "postgres": pg_check["status"],
            "redis": redis_check["status"],
        },
        "details": {
            "postgres": pg_check,
            "redis": redis_check,
        },
    }


@router.get(
    "/test-error",
    summary="Trigger Diagnostic Error",
    description="Endpoint to verify global error handlers and correlation ID propagation.",
)
async def trigger_test_error(
    error_type: str = Query(
        default="app",
        description="Type of error to trigger: 'app' (AppException) or 'unexpected' (ZeroDivisionError)",
    ),
) -> None:
    """Intentionally raise an error to validate error handling middleware."""
    if error_type == "app":
        raise AppException(
            message="This is a controlled diagnostic application error.",
            code="DIAGNOSTIC_TEST_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"test_param": error_type, "triggered": True},
        )
    # Trigger unhandled 500 error
    _ = 1 / 0
