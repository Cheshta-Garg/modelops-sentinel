import time
import uuid
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.errors import build_error_response
from app.core.logging import get_logger, request_id_ctx
from app.core.metrics import normalize_endpoint, track_http_request

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to inject correlation Request ID, log request lifecycle, and track Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract existing request ID or generate a new UUIDv4
        req_id = request.headers.get("X-Request-ID")
        if not req_id:
            req_id = uuid.uuid4().hex

        # Bind to contextvar for propagation across asynchronous coroutines
        token = request_id_ctx.set(req_id)
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration_s = time.perf_counter() - start_time
            duration_ms = round(duration_s * 1000, 2)

            # Record Prometheus metrics for HTTP traffic
            endpoint = normalize_endpoint(request.url.path)
            track_http_request(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code,
                duration_seconds=duration_s,
            )

            # Inject tracking headers
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Response-Time-Ms"] = str(duration_ms)

            logger.info(
                f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)"
            )
            return response

        except Exception as exc:
            duration_s = time.perf_counter() - start_time
            duration_ms = round(duration_s * 1000, 2)

            # Record metrics for 500 unhandled errors
            endpoint = normalize_endpoint(request.url.path)
            track_http_request(
                method=request.method,
                endpoint=endpoint,
                status_code=500,
                duration_seconds=duration_s,
            )

            logger.error(
                f"{request.method} {request.url.path} raised an unhandled exception after {duration_ms}ms: {exc}",
                exc_info=True,
            )
            response = build_error_response(
                status_code=500,
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred. Please contact system operators.",
                details=None,
            )
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Response-Time-Ms"] = str(duration_ms)
            return response

        finally:
            # Reset context variable to prevent leakage in reused threads/tasks
            request_id_ctx.reset(token)
