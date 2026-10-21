# Phase 04 — Synchronous ML Model Serving & Inference Engine

## Objective
Implement high-performance synchronous machine learning model inference for **ModelOps Sentinel**. Build an explicit, decoupled serving engine (`InferenceEngine`) providing:
1. **In-process memory caching** to eliminate expensive repeated disk I/O and deserialization overhead.
2. **Deterministic Redis result cache-aside** with canonical feature hashing.
3. **Sub-millisecond latency tracking** using high-resolution monotonic timers.
4. **End-to-end correlation tracing** via request IDs.
5. **Telemetry event creation** with PostgreSQL audit persistence and non-blocking Redis buffering (`telemetry:buffer`).
6. **Graceful fail-open resilience** ensuring that Redis connectivity failures never degrade user-facing prediction responses.

---

## Why This Stage Exists
In production ML systems, naïve serving architectures often make fatal design errors:
1. **Per-Request Model Deserialization**: Loading model binaries from disk (`joblib.load`) on every incoming HTTP request causes massive disk I/O, heavy CPU overhead, memory fragmentation, and latency spikes (50–300ms overhead per call).
2. **Coupling Serving Logic to API Handlers**: Blending data validation, tensor formatting, prediction execution, caching, and telemetry directly inside FastAPI route handlers makes code untestable, unmaintainable, and impossible to reuse across worker queues or batch endpoints.
3. **Absence of Cache-Aside for High-Volume Queries**: Real-world prediction pipelines often receive repeated requests with identical feature vectors (e.g. repeated IoT sensor polling, web traffic bots, or status checks). A Redis cache-aside layer eliminates redundant matrix multiplication and tree traversals.
4. **Lack of Telemetry for Drift & Auditability**: Without recording every served prediction (inputs, outputs, latency, timestamps), downstream drift monitoring, model rollback, and accuracy evaluation against ground-truth labels are impossible.
5. **Hard Dependencies on Cache Subsystems**: If Redis goes down, a poorly designed inference engine crashes all prediction requests. Production-grade systems must fail open.

---

## What Was Implemented
- **Baseline SWaT Anomaly Detection Model (`scripts/train_baseline_model.py`)**:
  - Secure Water Treatment (SWaT) physical sensor telemetry model (`LIT101`, `FIT101`, `AIT201`, `P101`, `DPIT301`).
  - Scikit-learn `RandomForestClassifier` trained and serialized via `joblib` to `artifacts/baseline_swat_model.joblib`.
- **Inference Engine (`app/services/inference.py`)**:
  - `InferenceEngine` class decoupling model execution from HTTP routing.
  - **In-process Model Cache**: Thread-safe in-memory dictionary caching deserialized model instances by `version_id`.
  - **Deterministic Feature Hashing**: Canonical serialization of features (sorted keys, type normalization, SHA-256 digest) to generate collisions-free Redis cache keys: `cache:inf:{model_id}:{version_tag}:{feature_hash}`.
  - **Redis Cache-Aside**: Check Redis for existing predictions (`TTL = 300s`); return immediately on cache hits with `cached=True`.
  - **Feature Formatting & Validation**: Validates dictionary and list payloads against `version.input_schema`, raising `422 Unprocessable Entity` on missing features.
  - **Monotonic High-Precision Latency Measurement**: Utilizes `time.perf_counter()` to capture true execution duration in milliseconds.
  - **Telemetry Creation**: Persists immutable audit records into PostgreSQL (`InferenceRequest` table) and buffers telemetry events into Redis list `telemetry:buffer` for downstream asynchronous worker ingestion.
  - **Fail-Open Resilience**: All Redis operations (cache read, cache write, telemetry queue) run inside protected try-except blocks, ensuring predictions succeed in-memory even if Redis is unreachable.
- **Async Redis Connection Manager (`app/core/redis.py`)**:
  - Singleton `get_redis_client()` with connection pooling and graceful shutdown.
- **Serving API Endpoint (`app/api/v1/endpoints/models.py`)**:
  - `POST /api/v1/models/{id}/predict` returning `{request_id, model, version, prediction, confidence, latency_ms, cached}`.
- **Comprehensive Automated Test Suite (`tests/test_inference.py`)**:
  - 10 new tests validating dictionary inputs, list inputs, missing features (422), missing models (404), missing versions (404), cache hit/miss transitions, cache invalidation, runtime execution errors (500), and Redis fail-open behavior (37/37 total project tests passing).

---

## Files Created/Modified

| File | Purpose |
| :--- | :--- |
| `scripts/train_baseline_model.py` | Training script generating SWaT anomaly detection model artifact. |
| `app/core/redis.py` | Global Redis client connection manager with connection pooling. |
| `app/schemas/inference.py` | Pydantic DTO contracts for inference requests and responses. |
| `app/services/inference.py` | Decoupled serving engine (`InferenceEngine`) with in-process & Redis caching. |
| `app/services/__init__.py` | Exported `InferenceEngine` and `ModelRegistryService`. |
| `app/api/v1/endpoints/models.py` | Added `POST /api/v1/models/{model_id}/predict` endpoint. |
| `tests/test_inference.py` | 10 unit and integration tests covering all serving paths. |
| `docs/06_phase_04_inference.md` | Comprehensive architectural documentation for Phase 04. |

---

## Tech Used
- **scikit-learn & joblib**: High-performance model training, evaluation, and binary deserialization.
- **NumPy**: Fast vectorized 2D array feature transformations.
- **FastAPI & Pydantic v2**: High-throughput asynchronous routing and strict payload schema validation.
- **Redis (aioredis)**: High-speed in-memory result caching and non-blocking telemetry queuing.
- **SQLAlchemy 2.0 Async & asyncpg**: Transactional persistence of prediction audit records.
- **Python contextvars**: Distributed request correlation tracking across coroutines.
- **Pytest & pytest-asyncio**: Unit testing of cache lifecycles, error paths, and engine resilience.

---

## Architecture / Flow

```
Client (HTTP POST /api/v1/models/{id}/predict)
       │
       ▼ [RequestContextMiddleware] -> Injects X-Request-ID, binds contextvar
FastAPI Route (models.py)
       │
       ▼ [InferenceEngine.predict()]
       │
       ├─▶ 1. Query PostgreSQL: Fetch Model & Target Version (ACTIVE or STAGING)
       │
       ├─▶ 2. Compute Deterministic Feature Cache Key:
       │       SHA-256(canonical JSON of features) -> cache:inf:{model_id}:{version}:{hash}
       │
       ├─▶ 3. Check Redis Result Cache:
       │       ├─ HIT: Return cached prediction immediately (cached = True)
       │       └─ MISS (or Redis down): Proceed to in-memory execution
       │
       ├─▶ 4. Validate & Format Input Features (422 if missing required keys)
       │
       ├─▶ 5. Fetch Model from In-Process Memory Cache (_model_cache):
       │       ├─ HIT: Use loaded model instance
       │       └─ MISS: joblib.load(abs_path) -> store in _model_cache
       │
       ├─▶ 6. Execute Model Inference:
       │       - model.predict(X)
       │       - model.predict_proba(X) -> confidence = max(probabilities)
       │       - Measure latency via time.perf_counter()
       │
       ├─▶ 7. Store Result in Redis Cache (TTL = 300s, fail-open)
       │
       ├─▶ 8. Telemetry Creation:
       │       - Persist InferenceRequest in PostgreSQL
       │       - LPUSH to Redis queue "telemetry:buffer" (fail-open)
       │
       └─▶ 9. Return HTTP 200 OK:
               {
                 "request_id": "...",
                 "model": "swat_anomaly_model",
                 "version": "v1.0.0",
                 "prediction": 0,
                 "confidence": 0.985,
                 "latency_ms": 3.42,
                 "cached": false
               }
```

---

## Important Concepts Explained

### 1. Model Loading
The operation of reading a serialized model file (e.g., `.joblib`, `.pkl`, `.onnx`) from secondary storage (SSD/HDD/network storage), allocating memory in Python's heap, reconstructing the object tree, and restoring trained weights.
- **Why it matters**: Deserialization is computationally expensive (unpickling complex trees, verifying class definitions, allocating memory). Doing it inside a request handler causes severe throughput degradation.
- **Production Pattern**: Models should only be loaded once per process lifecycle or on-demand on the first request, and then kept resident in RAM.

### 2. Model Cache (In-Process)
An application-level in-memory cache (`dict[str, Any]` in Python) that retains active model instances in RAM across HTTP requests within the serving process.
- **Why it matters**: Reading a model from RAM takes nanoseconds ($\approx 0.0001\text{ms}$), whereas reading and unpickling from disk takes $10\text{–}100\text{ms}$. In-process caching eliminates 99.9% of serving overhead for repeat requests.
- **Eviction Strategy**: When memory is constrained or a model version is retired/updated, the entry is evicted via `_model_cache.pop(version_id)`.

### 3. Cache Key Generation
A deterministic string representation uniquely identifying a set of input features for a given model version.
- **The Challenge**: In Python and JSON, dictionaries are key-order independent (`{"a": 1, "b": 2}` is logically identical to `{"b": 2, "a": 1}`). Floating-point numbers can also have varying string representations.
- **Our Solution**:
  1. Extract values in the canonical order defined by `version.input_schema["features"]`.
  2. Serialize using `json.dumps(..., sort_keys=True)`.
  3. Compute a SHA-256 digest truncated to 16 characters.
  4. Format: `cache:inf:{model_id}:{version_tag}:{feature_hash}`.

### 4. Cache Invalidation
The explicit removal or expiration of cached prediction results and loaded model instances when the underlying state changes.
- **Result Cache Invalidation**: Cached predictions have a Time-To-Live (TTL = 300s) to prevent serving stale data indefinitely. When a model version is rolled back or redeployed, keys under `cache:inf:{model_id}:{version_tag}:*` are purged.
- **Model Cache Invalidation**: When an operator deactivates or rolls back a version, calling `invalidate_model_cache(version_id)` forces the engine to clear the in-memory instance, ensuring no zombie model versions continue serving.

### 5. Inference Latency
The elapsed duration from when the serving engine receives the validated request payload to when the prediction and confidence score are computed.
- **Wall-Clock vs. Monotonic Timer**: Wall-clock timers (`time.time()`) are subject to NTP time synchronization jumps and leap seconds, which can report negative or inflated latencies. Monotonic timers (`time.perf_counter()`) guarantee strictly increasing, high-resolution fractional seconds unaffected by system clock adjustments.
- **SLA Composition**:
  $$\text{Total Latency} = \text{Network} + \text{Middleware} + \text{Validation} + \text{Inference} + \text{Serialization}$$

### 6. Synchronous Inference
A synchronous request-response communication pattern where the client opens an HTTP connection and blocks waiting for the server to calculate and return the prediction in the same HTTP cycle.
- **When to Use**: Interactive user requests, fraud detection at point of sale, real-time recommendation feeds, and anomaly detection where decisions must be made in $< 50\text{ms}$.
- **When NOT to Use**: Heavy computer vision models, video transcription, large LLM generation, or batch ETL over millions of rows (which require asynchronous message queues and worker pools).

---

## API Changes

### New Endpoint
`POST /api/v1/models/{model_id}/predict`

**Request Body**:
```json
{
  "features": {
    "LIT101": 650.0,
    "FIT101": 2.45,
    "AIT201": 255.0,
    "P101": 1,
    "DPIT301": 19.5
  },
  "version_tag": "v1.0.0"
}
```

**Response (200 OK)**:
```json
{
  "request_id": "c1f7b8a923d4e5f6",
  "model": "swat_anomaly_model",
  "version": "v1.0.0",
  "prediction": 0,
  "confidence": 0.985,
  "latency_ms": 3.74,
  "cached": false
}
```

---

## Testing

### Automated Test Results (Pytest)
```
collected 37 items

tests/test_database.py::test_all_12_tables_created PASSED                [  2%]
tests/test_database.py::test_create_user_and_model PASSED                [  5%]
tests/test_database.py::test_user_unique_email_constraint PASSED         [  8%]
tests/test_database.py::test_model_version_unique_tag_constraint PASSED  [ 10%]
tests/test_database.py::test_offline_metrics_and_reference_stats PASSED  [ 13%]
tests/test_database.py::test_deployment_canary_routing PASSED            [ 16%]
tests/test_database.py::test_inference_request_and_ground_truth PASSED   [ 18%]
tests/test_database.py::test_online_and_perf_metrics PASSED              [ 21%]
tests/test_database.py::test_drift_event_and_audit_log PASSED            [ 24%]
tests/test_database.py::test_cascade_delete_model_removes_children PASSED [ 27%]
tests/test_health.py::test_liveness_probe_returns_200 PASSED             [ 29%]
tests/test_health.py::test_request_id_middleware_generates_header PASSED [ 32%]
tests/test_health.py::test_request_id_middleware_preserves_client_header PASSED [ 35%]
tests/test_health.py::test_readiness_probe_disconnected_subsystems PASSED [ 37%]
tests/test_health.py::test_readiness_probe_connected_subsystems PASSED   [ 40%]
tests/test_health.py::test_handled_app_exception_flow PASSED             [ 43%]
tests/test_health.py::test_unhandled_exception_flow PASSED               [ 45%]
tests/test_inference.py::test_valid_inference_with_dict_features PASSED  [ 48%]
tests/test_inference.py::test_valid_inference_with_list_features PASSED  [ 51%]
tests/test_inference.py::test_inference_invalid_input_missing_features PASSED [ 54%]
tests/test_inference.py::test_inference_invalid_input_wrong_list_length PASSED [ 56%]
tests/test_inference.py::test_inference_missing_model_returns_404 PASSED [ 59%]
tests/test_inference.py::test_inference_missing_version_returns_404 PASSED [ 62%]
tests/test_inference.py::test_inference_cache_miss_then_cache_hit PASSED [ 64%]
tests/test_inference.py::test_inference_cache_invalidation PASSED        [ 67%]
tests/test_inference.py::test_inference_execution_error_handled PASSED   [ 70%]
tests/test_inference.py::test_inference_redis_fail_open_resilience PASSED [ 72%]
tests/test_registry.py::test_register_model_success PASSED               [ 75%]
tests/test_registry.py::test_register_model_duplicate_name_conflict PASSED [ 78%]
tests/test_registry.py::test_list_models PASSED                          [ 81%]
tests/test_registry.py::test_get_model_by_id_and_not_found PASSED        [ 83%]
tests/test_registry.py::test_upload_valid_joblib_artifact_pipeline PASSED [ 86%]
tests/test_registry.py::test_upload_version_duplicate_tag_conflict PASSED [ 89%]
tests/test_registry.py::test_upload_invalid_extension_rejected PASSED    [ 91%]
tests/test_registry.py::test_upload_corrupted_model_fails_validation PASSED [ 94%]
tests/test_registry.py::test_upload_artifact_without_predict_method_fails PASSED [ 97%]
tests/test_registry.py::test_upload_malformed_schema_json_fails PASSED   [100%]

============================= 37 passed in 37.38s =============================
```

### Linter Results (Ruff)
```
.\.venv\Scripts\ruff.exe check .
All checks passed!
```

---

## Commands Used
```bash
# 1. Train the baseline SWaT anomaly detection model
.\.venv\Scripts\python.exe scripts/train_baseline_model.py

# 2. Run inference unit and integration tests
.\.venv\Scripts\pytest.exe tests/test_inference.py -v

# 3. Run entire test suite across all stages
.\.venv\Scripts\pytest.exe -v

# 4. Verify code quality and styling
.\.venv\Scripts\ruff.exe check .
```

---

## Expected Behavior
- Submitting valid named sensor features returns a prediction class (`0` for normal, `1` for attack), confidence score ($\ge 0.50$), execution latency in milliseconds, and correlation ID.
- Calling the endpoint twice with identical features results in a cache miss on request 1 (`cached: false`) and a cache hit on request 2 (`cached: true`) with lower latency.
- Omitting required features defined in the model schema immediately returns `422 Unprocessable Entity` with an explicit list of missing features.
- Requesting an unapproved version status (e.g. `FAILED` or `REGISTERED`) returns `400 Bad Request`.
- If Redis is disconnected, inference completes smoothly in-memory without erroring out (fail-open).

---

## Problems Encountered & Solutions
1. **Unused Imports in Test Module**:
   - *Problem*: `tests/test_inference.py` initially imported `get_redis_client` and `ModelVersion`, which were unused and flagged by Ruff (`F401`).
   - *Solution*: Removed unused imports and formatted using Ruff's automated fixer (`--fix`).
2. **Deterministic Feature Ordering**:
   - *Problem*: Python dictionaries in JSON requests can deliver keys in arbitrary orders, leading to cache misses for semantically identical inputs.
   - *Solution*: Normalizing feature ordering against `version.input_schema["features"]` ensures canonical, deterministic hashing across all client implementations.

---

## Decisions / Tradeoffs
- **Fail-Open Result Caching**: Rather than aborting inference when Redis experiences connection issues, the engine catches Redis exceptions, logs a warning, and executes in-memory prediction. In production, serving a fresh prediction with an extra 2ms latency is vastly preferable to returning an HTTP 500 error to users.
- **Separation of Concerns**: Kept the model serving engine (`InferenceEngine`) strictly independent of FastAPI routes. This enables reusing the same engine inside background worker tasks (`arq`) for asynchronous batch inference without HTTP overhead.

---

## Interview Questions

### Q1: Why should model loading and unpickling never occur inside an incoming HTTP request handler?
> **Answer**: Deserializing a model from disk via `joblib.load()` or `pickle.load()` requires blocking disk I/O, Python heap allocations, and object graph reconstruction. This takes between 10ms to several seconds depending on model size. Under concurrent traffic, repeatedly loading from disk causes extreme CPU thrashing, memory fragmentation, GIL contention, and catastrophic latency spikes. Models must be loaded once and cached in-process in RAM.

### Q2: What is the cache stampede (thundering herd) problem in ML inference caching, and how is it addressed?
> **Answer**: When a high-frequency cache key expires (e.g., TTL expires on a common sensor reading), dozens of concurrent requests simultaneously observe a cache miss and all trigger redundant, computationally heavy inference jobs at the exact same moment. This can be addressed by:
> 1. Mutual exclusion locks (mutex/single-flight) around cache misses so only one worker computes while others wait.
> 2. Probabilistic early expiration (XFetch algorithm) where requests randomly refresh the cache before the TTL expires.

### Q3: Why is high-resolution monotonic time (`time.perf_counter()`) required instead of wall-clock time (`time.time()`) for latency tracking?
> **Answer**: `time.time()` reads the system wall-clock, which can be modified backward or forward by NTP (Network Time Protocol) synchronization or leap seconds. This can produce inaccurate, zero, or even negative latency metrics. `time.perf_counter()` reads the CPU's monotonic hardware clock, which is guaranteed to always move forward at a constant rate, providing sub-millisecond precision.

---

## Current Project State
- Stage 4 complete.
- Synchronous ML inference serving engine implemented and operational.
- In-process model caching and Redis result cache-aside working.
- Telemetry buffer (`telemetry:buffer`) and database audit logging active.
- 37/37 automated tests passing across all four stages.
- Linter clean: 0 errors.

---

## Next Stage
**Stage 5: Deployment Strategies (Direct, Canary & Shadow Traffic Routing)**
- Implementation of `deployments` management.
- Dynamic traffic splitting between baseline and candidate model versions.
- Safe canary routing and rollback mechanisms.
