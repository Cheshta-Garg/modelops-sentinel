# Phase 06 — Asynchronous ML Inference, Background Workers & Redis Queuing (arq)

## Objective
Implement high-throughput, decoupled asynchronous machine learning inference and background telemetry aggregation for **ModelOps Sentinel**.
Introduce an asynchronous distributed queue architecture leveraging **FastAPI**, **Redis**, and **arq** to ensure:
1. Immediate non-blocking ingestion of heavy prediction requests returning `202 Accepted`.
2. Decoupled worker execution isolating intensive model computation from ASGI web server event loops.
3. Resilience against transient failures via automatic retries with exponential/fixed backoff.
4. Protection against zombie tasks and infinite hanging via explicit job timeouts.
5. Non-blocking telemetry ingestion and batch summarization.
6. Deterministic dual-mode operation: running on distributed `arq` Redis in production, with an in-memory async worker fallback for Zero-Docker / testing environments.

---

## Why This Stage Exists
While synchronous inference (`POST /models/{id}/predict`) is suitable for sub-50ms predictions on lightweight tabular models, real-world ModelOps deployments face operational realities that require asynchronous job queues:
1. **Preventing ASGI Thread & Socket Exhaustion**: Synchronous requests hold open client HTTP connections throughout feature validation, preprocessing, and model execution. Under burst traffic, connection pools saturate, rejecting incoming traffic.
2. **Eliminating Reverse-Proxy Gateway Timeouts**: Heavy inference pipelines (e.g. high-dimensional sensors, tree ensembles, computer vision, or NLP transforms) can exceed 30–60 second gateway timeouts on load balancers, causing HTTP `504 Gateway Timeout` errors even when the model finishes successfully.
3. **Queue-Based Load Leveling (Backpressure)**: A queue acts as a shock absorber. When traffic spikes 10x, the queue depth expands, allowing workers to process tasks at a steady, sustainable rate rather than crashing the API server with Out-Of-Memory (OOM) errors.
4. **Decoupled Telemetry Aggregation**: Telemetry persistence, latency percentiles, and online metrics calculation should not add latency to the client's request path.

---

## Core Theoretical Concepts Explained

### 1. Why Asynchronous Processing?
* **Decoupling Latency from Ingestion**: The API layer's sole responsibility is accepting the payload, validating authentication and schema bounds, persisting the job to the queue, and returning immediately (<5ms).
* **Workload Isolation**: CPU-intensive vector transformations, tensor operations, and model inference run in dedicated worker processes, leaving the API event loop completely unblocked to handle high-concurrency I/O.

### 2. Queue (Task Broker)
* **Definition**: A durable, ordered buffer (implemented in Redis via sorted sets and lists in `arq`) that stores serialized job instructions awaiting worker execution.
* **Backpressure Management**: Enables smooth degradation under high load. Jobs wait in the queue without consuming CPU or memory on the API server.
* **Decoupled Availability**: The API can continue enqueuing requests even if workers are temporarily restarting, scaling up, or deploying an updated runtime.

### 3. Worker (Dedicated Execution Engine)
* **Definition**: An independent daemon process (`arq app.worker.settings.WorkerSettings`) that continuously pulls tasks from the Redis queue, initializes execution contexts, runs the job, and writes back status and results.
* **Independent Scalability**: In Kubernetes or container environments, web replicas can scale based on HTTP request counts, while worker pods scale based on CPU/GPU utilization or queue depth (`arq:queue` length).

### 4. Retry & Backoff
* **Fault Tolerance**: Network blips, temporary database connection limits, or lock contention are transient. Retries automatically re-attempt failed jobs with a backoff delay (`retry_delay = 2s`), distinguishing temporary blips from fatal bugs without manual intervention.
* **Configured Safety**: Tasks are bounded by `max_tries = 3`. If all attempts fail, the job transitions permanently to `failed`, recording the exception details in the job registry.

### 5. Timeout
* **Circuit Breaker for Execution**: Every task has an enforceable upper bound (`job_timeout = 60s`).
* **Starvation Prevention**: If a task hangs due to an infinite loop, external socket hang, or deadlocked thread, the worker terminates the execution, preventing worker starvation and freeing up capacity for pending tasks.

### 6. Eventual Consistency
* **Contract**: Rather than offering immediate synchronous consistency, the system guarantees that the system state will converge to the correct final result after a finite delay.
* **Application**: The client receives a ticket (`job_id`) upon submission. The prediction is computed asynchronously, and the final state (`complete` or `failed`) becomes visible upon subsequent status polling (`GET /api/v1/jobs/{job_id}`).

### 7. HTTP 202 Accepted Response
* **RFC 9110 Standard**: Signals to the client that the request has been received, accepted, and queued for processing, but processing has not yet completed.
* **Payload Contract**:
  ```json
  {
    "job_id": "b9f71c3e-8f24-4f51-bca2-892c57f201e8"
  }
  ```
* Provides the client with a reference URI or job ID to check progress asynchronously.

---

## Architecture & Workflows

### 1. Asynchronous Inference Enqueue Flow

```
Client                        FastAPI API Route                   Redis Queue
  │                                   │                                │
  ├─── POST /models/{id}/predict/async▶│                                │
  │    {features, version_tag}        ├─▶ Validate Model & Version     │
  │                                   ├─▶ Check Auth & RBAC            │
  │                                   ├─▶ queue_service.enqueue() ────▶│ [LPUSH arq:queue]
  │                                   │   (generates job_id)           │ (stores task & args)
  │◀── 202 Accepted {job_id} ─────────┤                                │
```

### 2. Background Worker Execution & Result Recording

```
arq Worker                         Redis Queue                     Database / Storage
    │                                   │                                  │
    ├─── Poll for next available job ──▶│                                  │
    │◀── Pulls async_predict_task ──────┤                                  │
    │                                                                      │
    ├─▶ 1. Load Model Version from In-Process Cache or Disk                │
    ├─▶ 2. Format & Validate Features                                      │
    ├─▶ 3. Execute model.predict(X) & predict_proba(X)                     │
    ├─▶ 4. Measure Latency (time.perf_counter)                             │
    ├─▶ 5. Write Audit & Telemetry Record ────────────────────────────────▶│ (InferenceRequest)
    │                                                                      │
    ├─▶ 6. Write Result & Status to Redis ────────────────────────────────▶│ [SET arq:result:job_id]
```

### 3. Job Status Polling Flow

```
Client                        FastAPI (/jobs/{job_id})            Redis / Memory Queue
  │                                      │                                 │
  ├─── GET /api/v1/jobs/{job_id} ───────▶│                                 │
  │    Header: Bearer <jwt>              ├─▶ queue_service.get_job() ─────▶│
  │                                      │◀─ Return JobStatusResponse ─────┤
  │◀── 200 OK {status, result, error} ───┤                                 │
```

---

## What Was Implemented

1. **Job Contracts & DTOs (`app/schemas/job.py`)**:
   * `AsyncJobResponse`: Returns `{ "job_id": "..." }` on HTTP 202.
   * `JobStatusResponse`: Full lifecycle contract exposing `status`, `function`, `enqueue_time`, `start_time`, `finish_time`, `success`, `result`, `error`, and `retries`.
2. **Worker Tasks (`app/worker/tasks.py`)**:
   * `async_predict_task`: Background prediction task executing model inference, calculating confidence and latency, and persisting records into the `InferenceRequest` database table. Supports `Retry` on transient faults and records errors.
   * `process_telemetry_task`: Background telemetry aggregator calculating batch metrics (event counts, average latency, prediction class distribution).
   * `TASK_REGISTRY`: Mapping linking task names to executable worker coroutines.
3. **Worker Settings (`app/worker/settings.py`)**:
   * `WorkerSettings`: Configuration for arq worker daemon defining Redis connection settings, `job_timeout = 60s`, `max_tries = 3`, `retry_delay = 2s`, and startup/shutdown lifecycle hooks.
4. **Unified Queue Service (`app/services/queue.py`)**:
   * `QueueService`: High-level singleton facade managing task submission and status retrieval.
   * **Production Arq Backend**: Interfaces directly with Redis via `arq.create_pool` and `arq.jobs.Job`.
   * **In-Memory Worker Fallback (`MemoryQueueBackend`)**: Zero-Docker in-memory async worker engine replicating arq lifecycle states, timeouts (`asyncio.wait_for`), retries (`Retry`), and restart simulation.
   * **Circuit Breaker**: Detects offline Redis and falls back instantly without incurring network connection timeout latency.
5. **API Endpoints**:
   * `POST /api/v1/models/{model_id}/predict/async` (`app/api/v1/endpoints/models.py`): Accepts inference payload, verifies active/staging version, and enqueues job returning HTTP 202 Accepted.
   * `GET /api/v1/jobs/{job_id}` (`app/api/v1/endpoints/jobs.py`): Fetches live job status, progress, results, or error traces.
   * Registered `jobs.router` in `app/api/v1/router.py`.

---

## Files Created / Modified

| File | Type | Description |
| :--- | :---: | :--- |
| `requirements.txt` | Modified | Added `arq>=0.28.0` and pinned `redis>=5.0.0,<6.0.0`. |
| `app/schemas/job.py` | Created | Pydantic contracts for `AsyncJobResponse` and `JobStatusResponse`. |
| `app/worker/__init__.py` | Created | Worker package initialization. |
| `app/worker/tasks.py` | Created | Asynchronous inference and telemetry processing worker tasks. |
| `app/worker/settings.py` | Created | Arq `WorkerSettings` configuration with timeouts and retry policies. |
| `app/services/queue.py` | Created | `QueueService` with Redis `arq` backend and `MemoryQueueBackend` fallback. |
| `app/services/__init__.py` | Modified | Exported `QueueService` and `queue_service`. |
| `app/api/v1/endpoints/jobs.py` | Created | Route handler for `GET /api/v1/jobs/{job_id}`. |
| `app/api/v1/endpoints/models.py` | Modified | Added `POST /api/v1/models/{model_id}/predict/async` returning HTTP 202. |
| `app/api/v1/router.py` | Modified | Included `jobs.router` in API v1 prefix. |
| `tests/test_worker.py` | Created | 12 automated unit and integration tests covering the complete async worker lifecycle. |
| `scripts/test_stage6_manual.py` | Created | Zero-Docker manual verification script demonstrating async inference, polling, timeouts, and retries. |
| `docs/08_phase_06_worker.md` | Created | Comprehensive architectural documentation for Phase 06. |

---

## Verification & Testing

### 1. Automated Test Suite (61/61 Tests Passing)
Run the full test suite across the entire project:
```bash
.\.venv\Scripts\python.exe -m pytest -v
```
Output:
```
tests/test_auth.py::... PASSED                                           [ 19%]
tests/test_database.py::... PASSED                                       [ 36%]
tests/test_health.py::... PASSED                                         [ 47%]
tests/test_inference.py::... PASSED                                      [ 63%]
tests/test_registry.py::... PASSED                                       [ 80%]
tests/test_worker.py::test_async_predict_returns_202_accepted PASSED     [ 81%]
tests/test_worker.py::test_async_predict_execution_and_status_polling PASSED [ 83%]
tests/test_worker.py::test_async_predict_anomaly_detection PASSED        [ 85%]
tests/test_worker.py::test_async_predict_unauthenticated_returns_401 PASSED [ 86%]
tests/test_worker.py::test_async_predict_missing_model_returns_404 PASSED [ 88%]
tests/test_worker.py::test_async_predict_missing_version_tag_returns_404 PASSED [ 90%]
tests/test_worker.py::test_async_predict_invalid_features_records_failure_state PASSED [ 91%]
tests/test_worker.py::test_get_job_not_found_returns_404 PASSED          [ 93%]
tests/test_worker.py::test_process_telemetry_task PASSED                 [ 95%]
tests/test_worker.py::test_worker_task_retry_behavior PASSED             [ 96%]
tests/test_worker.py::test_worker_task_timeout_behavior PASSED           [ 98%]
tests/test_worker.py::test_worker_restart_simulation PASSED              [100%]
======================= 61 passed in 50.77s =======================
```

### 2. Manual Zero-Docker Verification Script
Run the manual verification script:
```bash
.\.venv\Scripts\python.exe scripts/test_stage6_manual.py
```
**Verified Behaviors**:
- Initialized database schema.
- Authenticated Engineer user and acquired JWT token.
- Ensured SWaT baseline model and version are staged.
- Submitted normal inference: received `HTTP 202 Accepted` with `job_id`.
- Polled `/jobs/{job_id}`: transitioned from `queued` to `complete`, returning `prediction: 0` (normal process).
- Submitted cyber-attack inference: polled until completion, returning `prediction: 1` (cyber-attack anomaly detected).
- Tested failure state handling: submitted missing features, job transitioned to `status: failed`, capturing validation error details.
- Tested telemetry processing task: aggregated batch event count, average latency, and class distribution.
- Tested retries: worker transient failure re-attempted and succeeded on retry.
- Tested timeouts: worker terminated hanging task after 0.2s, marking job as `failed`.
- Tested worker restart: preserved queued jobs and verified state recovery.

---

## Interview Defense / Technical Questions & Answers

### Q1: Why choose `arq` over Celery for ModelOps Sentinel?
**Answer**:
Celery is a legacy, heavyweight task queue built around synchronous Python and multiprocessing. `arq` is a modern, lightweight, fully asynchronous task queue built natively for Python's `asyncio` and Redis. Because ModelOps Sentinel is built on FastAPI and SQLAlchemy 2.0 Async, `arq` allows worker tasks to run natively on the async event loop, sharing connection pools without the serialization overhead, threading conflicts, or dependency bloat of Celery.

### Q2: How does the system prevent a long-running prediction from blocking the worker indefinitely?
**Answer**:
Every task has an enforced `job_timeout` configured in `WorkerSettings` (default 60s) and passed to `asyncio.wait_for`. If model execution exceeds this limit (e.g. infinite loops in custom preprocessing or CUDA deadlocks), the task is cancelled, an `asyncio.TimeoutError` is logged, and the job transitions to `failed` with an explicit timeout error message.

### Q3: What is the purpose of returning HTTP 202 Accepted instead of 200 OK?
**Answer**:
HTTP 200 OK implies the requested operation has finished and its full output is contained in the response. HTTP 202 Accepted is the standardized HTTP status code (RFC 9110) indicating that the request has been validated and queued for asynchronous processing, but execution is not yet complete. It sets the architectural expectation that the client should poll the status endpoint (`GET /api/v1/jobs/{job_id}`) or listen for completion.
