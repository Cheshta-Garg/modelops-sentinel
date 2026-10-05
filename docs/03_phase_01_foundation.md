# Phase 01 — Project Foundation & Core Infrastructure

## Objective
Establish the foundational software engineering scaffolding for **ModelOps Sentinel**. Build a clean, reproducible, production-oriented repository containing configuration management, structured logging, request tracing, container definitions, health probes, and testing/linting suites before implementing business domain logic.

---

## Why This Stage Exists
In enterprise software engineering, jumping straight into business logic without foundational infrastructure leads to technical debt: untraceable bugs, ad-hoc configurations, unhandled 500 errors, and non-reproducible environments. 

This stage establishes:
1. **Config Determinism**: Environment variables strictly validated through typed Pydantic models.
2. **Observability Scaffolding**: Correlation IDs (`X-Request-ID`) bound to async execution contexts so every log message can be traced back to an originating HTTP request.
3. **Container Readiness**: Explicit multi-stage Dockerfile and Docker Compose manifests specifying service boundaries and health dependencies.
4. **Reliability Gates**: Distinct liveness (`/health`) and readiness (`/ready`) endpoints to enable container orchestrators to make informed routing and restart decisions.

---

## What Was Implemented
- **Repository Structure**: Clean modular layout separating `api/`, `core/`, `tests/`, and `docs/`.
- **Configuration Management**: Type-safe settings using `pydantic-settings` reading from `.env`.
- **FastAPI Core**: Application factory pattern with lifespan management.
- **Middleware**: `RequestContextMiddleware` extracting or generating `X-Request-ID`, timing latency, and injecting response headers.
- **Structured Logging**: Dual formatters (human-readable text for local development, JSON for production log aggregators) pulling `request_id` from Python `contextvars`.
- **Global Error Handling**: Custom exception hierarchy (`AppException`, `NotFoundError`, `ServiceUnavailableError`) with standardized JSON error responses and safe fallback for unexpected 500s.
- **Health Probes**:
  - `/api/v1/health` (Liveness): Validates web worker event loop health.
  - `/api/v1/ready` (Readiness): Validates upstream PostgreSQL and Redis connectivity asynchronously.
  - `/api/v1/test-error` (Diagnostics): Endpoint to verify error handlers and request ID propagation.
- **Testing & Quality Tooling**: Pytest configuration with async fixtures, 7 comprehensive tests, and Ruff linting rules.
- **Container Infrastructure**: Multi-stage production `Dockerfile` with a non-root user and `docker-compose.yml` declaring `postgres`, `redis`, and `api` services.

---

## Files Created/Modified

| File | Purpose |
| :--- | :--- |
| `.gitignore` | Ignores `.venv`, Python bytecode, `.env`, artifacts, and test caches. |
| `.env.example` | Template documenting all environment variables with sensible defaults. |
| `.env` | Local development environment configuration. |
| `pyproject.toml` | Ruff linting and Pytest async configuration. |
| `requirements.txt` | Pinned foundational dependencies. |
| `Dockerfile` | Multi-stage build (builder $\rightarrow$ slim runner) running as non-root `appuser`. |
| `docker-compose.yml` | Multi-container setup for PostgreSQL 16, Redis 7, and the API service. |
| `README.md` | Project overview, architecture summary, and quickstart commands. |
| `app/__init__.py` | Package version marker. |
| `app/core/__init__.py` | Core infrastructure package marker. |
| `app/core/config.py` | Pydantic Settings model with computed DB and Redis URLs. |
| `app/core/logging.py` | JSON/Text formatters and `request_id_ctx` context variable management. |
| `app/core/errors.py` | Custom exception hierarchy and FastAPI global exception handlers. |
| `app/core/middleware.py` | Starlette middleware for Request ID injection and latency logging. |
| `app/api/__init__.py` | API package marker. |
| `app/api/v1/__init__.py` | V1 API package marker. |
| `app/api/v1/router.py` | V1 root router registering endpoint routers. |
| `app/api/v1/endpoints/__init__.py` | Endpoints package marker. |
| `app/api/v1/endpoints/health.py` | Liveness, readiness, and diagnostic error endpoints. |
| `app/main.py` | Application factory, middleware registration, and lifespan hooks. |
| `tests/__init__.py` | Test package marker. |
| `tests/conftest.py` | Pytest fixtures providing async `httpx.AsyncClient`. |
| `tests/test_health.py` | 7 automated tests covering liveness, readiness, tracing, and errors. |
| `docs/03_phase_01_foundation.md` | This phase documentation. |

---

## Tech Used
- **Python 3.10+**: Core programming language.
- **FastAPI**: Asynchronous web framework.
- **Pydantic & Pydantic-Settings**: Environment parsing and type validation.
- **asyncpg**: Asynchronous driver for PostgreSQL connectivity checks.
- **redis.asyncio**: Asynchronous Redis client for ping checks.
- **Starlette**: Base HTTP middleware and ASGI foundations.
- **Ruff**: High-speed Rust-based Python linter and formatter.
- **Pytest & pytest-asyncio**: Asynchronous test runner.
- **httpx**: ASGI async HTTP client for integration tests.
- **Docker & Docker Compose**: Multi-container specification.

---

## Architecture / Flow

```
Client Request
      │ (Optional Header: X-Request-ID)
      ▼
RequestContextMiddleware
      │ 1. Extract or generate UUID4 request_id
      │ 2. Set contextvars.ContextVar("request_id")
      │ 3. Start high-resolution timer (time.perf_counter)
      ▼
FastAPI Routing & Exception Handlers
      │
      ├── GET /api/v1/health
      │     └─▶ Return 200 OK {"status": "healthy"}
      │
      ├── GET /api/v1/ready
      │     ├─▶ Ping Postgres (asyncpg.connect)
      │     ├─▶ Ping Redis (client.ping)
      │     └─▶ Return 200 OK or 503 Service Unavailable
      │
      └── Exception Triggered (/api/v1/test-error)
            ├─▶ Handled: AppException -> Custom JSON (4xx)
            └─▶ Unhandled: Exception -> Safe 500 JSON + Traceback in logs
      │
      ▼
RequestContextMiddleware (Response Phase)
      │ 1. Calculate duration_ms
      │ 2. Set response header "X-Request-ID"
      │ 3. Set response header "X-Response-Time-Ms"
      │ 4. Log structured message: "[<request_id>] GET /path -> 200 (1.2ms)"
      │ 5. Reset contextvars token (prevent memory/task leak)
      ▼
Client Receives Response
```

---

## Important Concepts

### 1. Liveness vs. Readiness Probes
- **Liveness (`/health`)**: Answers *"Is the application process alive and responsive to HTTP traffic?"* If this fails, the orchestrator (Kubernetes, Docker) restarts the container.
- **Readiness (`/ready`)**: Answers *"Is the application ready to handle user requests?"* It verifies downstream dependencies (PostgreSQL, Redis). If this fails, the container is **not** killed; instead, the load balancer temporarily stops routing client traffic to it until it recovers.

### 2. Request Correlation IDs & `contextvars`
In an asynchronous event loop, a single thread interleaves the execution of hundreds of concurrent coroutines. Standard thread-local storage (`threading.local`) fails in async code because multiple requests run on the same thread. Python's `contextvars` module solves this by providing context-local storage that follows asynchronous task execution boundaries.

### 3. Graceful Error Normalization
Production APIs must never leak raw internal tracebacks, SQL statements, or file paths to end clients. The global error handler catches all unhandled exceptions, logs the full stack trace internally under the matching `request_id`, and returns a sanitized, standardized JSON payload to the user.

---

## API / DB Changes

### Endpoints Created
- `GET /api/v1/health`: Liveness probe (200 OK).
- `GET /api/v1/ready`: Readiness probe (200 OK if Postgres and Redis connected; 503 Service Unavailable if disconnected).
- `GET /api/v1/test-error`: Diagnostic endpoint triggering controlled (`400`) or unexpected (`500`) exceptions.
- `GET /`: Redirects to interactive Swagger documentation (`/docs`).

### Database Changes
*None in this stage (Database models and migrations scheduled for Stage 2).*

---

## Testing

### Automated Test Results (Pytest)
```
collected 7 items

tests/test_health.py::test_liveness_probe_returns_200 PASSED            [ 14%]
tests/test_health.py::test_request_id_middleware_generates_header PASSED [ 28%]
tests/test_health.py::test_request_id_middleware_preserves_client_header PASSED [ 42%]
tests/test_health.py::test_readiness_probe_disconnected_subsystems PASSED [ 57%]
tests/test_health.py::test_readiness_probe_connected_subsystems PASSED   [ 71%]
tests/test_health.py::test_handled_app_exception_flow PASSED            [ 85%]
tests/test_health.py::test_unhandled_exception_flow PASSED              [100%]

============================== 7 passed in 2.12s ==============================
```

### Linter Results (Ruff)
```
All checks passed!
```

### Live Endpoint Verifications
- **`/health`**: Returned `200 OK` with JSON body and `X-Request-ID` header.
- **`/ready`**: Returned `503 Service Unavailable` with `postgres: disconnected` and `redis: disconnected` when external services were offline.
- **`X-Request-ID`**: Verified client header `sentinel-client-trace-777` was retained and returned.
- **`/test-error?error_type=app`**: Returned `400 Bad Request` with structured error code `DIAGNOSTIC_TEST_ERROR`.
- **`/test-error?error_type=unexpected`**: Returned `500 Internal Server Error` with `INTERNAL_SERVER_ERROR` and matching `request_id`.

---

## Commands Used
```bash
# 1. Environment creation & dependency install
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# 2. Linting
.\.venv\Scripts\ruff check .

# 3. Test execution
.\.venv\Scripts\pytest

# 4. Start local development server
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## Expected Behavior
- Server starts up with a clean banner logging project name and environment.
- Any request made without an `X-Request-ID` receives a newly generated UUIDv4 in both response headers and structured logs.
- Client-provided `X-Request-ID` headers are faithfully echoed back.
- If PostgreSQL or Redis are not running, `/health` remains `200 OK` while `/ready` returns `503`.

---

## Problems Encountered & Solutions
1. **Starlette BaseHTTPMiddleware Unhandled Exception Bubble-Up**:
   - *Problem*: In Starlette, an unhandled exception inside an endpoint bubbles out of `BaseHTTPMiddleware` before the router's global `Exception` handler can format it into a response for the test client.
   - *Solution*: Added explicit `try...except Exception` handling inside `RequestContextMiddleware.dispatch` that catches unhandled errors, logs the full traceback with `request_id`, and directly returns `build_error_response(500, "INTERNAL_SERVER_ERROR")` with tracking headers attached.

---

## Decisions / Tradeoffs
- **Custom ContextVar Middleware vs. Third-Party Library**: Built lightweight in-house `RequestContextMiddleware` using standard library `contextvars` rather than pulling in external tracing packages. Eliminates dependencies and keeps the mechanism 100% transparent and explainable.
- **Native asyncpg & redis.asyncio in Readiness Probe**: Used direct async driver calls with strict 2-second timeouts rather than full ORM queries for the readiness check. Ensures the probe is ultra-lightweight and will not hang the health checker during network partitioning.

---

## Interview Questions

### Q1: Why must Kubernetes have separate Liveness and Readiness probes?
> **Answer**: If an app only has a single health probe and marks itself unhealthy when its database is temporarily unreachable, Kubernetes will restart the app container. Restarting an app that cannot reach its database does not fix the database; it creates a "crash loop" and puts extra restart load on the host. With separate probes, liveness stays healthy (preventing useless container restarts) while readiness returns 503, causing the load balancer to cleanly stop routing traffic until the database recovers.

### Q2: Why is `threading.local` dangerous in an asynchronous Python web framework like FastAPI?
> **Answer**: In synchronous multi-threaded frameworks (like Flask on Gunicorn), each concurrent request is assigned its own dedicated OS thread, so thread-local storage safely isolates request data. In FastAPI (ASGI), hundreds of concurrent requests run concurrently on a single thread's event loop via cooperative multitasking. If you store a `request_id` in `threading.local()`, coroutines will overwrite each other's data across `await` points. Python's `contextvars` provides context-local storage that isolates coroutine contexts regardless of the thread they execute on.

### Q3: Why is a multi-stage Docker build standard practice in production?
> **Answer**: Compiling dependencies requires build tools (e.g., `gcc`, `build-essential`, header files), which significantly inflate image size and introduce security attack surfaces (package managers, compilers). Multi-stage builds compile artifacts in a temporary `builder` image and copy only the final installed wheels/binaries into a lean `runner` image (e.g., `python:3.10-slim`), resulting in smaller images, faster deployment pulls, and minimized vulnerability footprints.

---

## Current Project State
- Foundation phase complete.
- Core FastAPI app, config, logging, middleware, health endpoints, tests, and container configs verified.
- Tests passing: 7/7.
- Linting clean: 0 errors.

---

## Next Stage
**Stage 2: Database Layer & Data Modeling**
- Define SQLAlchemy 2.0 async engine and sessionmaker.
- Create base model and core database models (`User`, `APIKey`, `Model`, `ModelVersion`).
- Configure Alembic for async migrations.
- Write initial migration revision.
- Implement database session dependency for route injection.
