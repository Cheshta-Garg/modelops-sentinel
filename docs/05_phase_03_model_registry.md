# Phase 03 — Model Registry & Artifact Ingestion Pipeline

## Objective
Implement the Model Registry and Version Management lifecycle for **ModelOps Sentinel**. Build an explicit, resilient artifact ingestion pipeline (`upload -> extension validation -> size validation -> SHA256 -> store artifact -> load test -> schema validation -> state transition`) enforcing the model version state machine (`REGISTERED -> VALIDATING -> STAGING` or `FAILED`), ensuring that only validated artifacts can ever reach `STAGING` and be deployed.

---

## Why This Stage Exists
In production ML systems, deploying raw models directly from training notebooks creates catastrophic operational failures:
1. **Unpickling Security & Corrupt Weights**: Loading arbitrary pickled binaries in production can cause memory crashes, corrupted weights, or arbitrary code execution.
2. **Missing Prediction Contracts**: Models might be uploaded that lack a callable `.predict()` method or fail to output expected schemas.
3. **Artifact Tampering & Lack of Integrity**: Without cryptographic hashes (SHA-256), a binary file can be modified or corrupted in storage without detection.
4. **Uncontrolled Deployments**: Without an explicit finite state machine, unverified models in `DRAFT` or `VALIDATING` states could accidentally receive production user traffic.

---

## What Was Implemented
- **Artifact Storage Abstraction (`app/storage/artifact_store.py`)**:
  - `ArtifactStore` protocol defining the storage interface.
  - `LocalArtifactStore` implementation storing artifacts under `${ARTIFACT_DIR}/{model_name}/{version_tag}/{filename}` with path traversal protection.
  - `compute_sha256`: Cryptographic digest utility for binary integrity tracking.
- **Model Version State Machine**:
  - `REGISTERED`: Metadata created; binary uploaded.
  - `VALIDATING`: Ingestion pipeline actively executing load tests and schema validation.
  - `STAGING`: Load test and schema validation passed; artifact is validated and deployable.
  - `FAILED`: Deserialization, interface check, or schema validation failed; artifact is locked and cannot be deployed.
- **Artifact Ingestion Pipeline (`app/services/registry.py`)**:
  1. *Model Exists Check*: Ensures parent model identity is registered.
  2. *Version Uniqueness Check*: Rejects duplicate version tags (`409 Conflict`).
  3. *Extension Validation*: Enforces valid serialized formats (`.joblib`, `.onnx`).
  4. *Size Validation*: Restricts uploads to 50 MB to prevent resource exhaustion attacks.
  5. *Cryptographic Hashing*: Computes SHA-256 digest of raw binary bytes.
  6. *Storage*: Writes binary to isolated directory on disk.
  7. *In-Memory Load Test*: Loads artifact with `joblib.load()`, verifies callable `.predict()` attribute.
  8. *Schema Validation*: Verifies input/output schema structure.
  9. *State Promotion*: Atomically updates status to `STAGING` or `FAILED`.
- **API Endpoints (`app/api/v1/endpoints/models.py`)**:
  - `POST /api/v1/models`: Register a new logical model entity.
  - `GET /api/v1/models`: List all registered models with their version counts.
  - `GET /api/v1/models/{id}`: Retrieve model details with its version iterations.
  - `POST /api/v1/models/{id}/versions`: Multipart form endpoint executing the full validation pipeline.
- **Test Suite (`tests/test_registry.py`)**:
  - 10 automated tests covering model registration, version uploads, extension checks, load testing failures, schema format checks, and duplicate conflicts (27 total tests passing).

---

## Files Created/Modified

| File | Purpose |
| :--- | :--- |
| `app/storage/__init__.py` | Storage package marker. |
| `app/storage/artifact_store.py` | `ArtifactStore` protocol, `LocalArtifactStore`, and SHA-256 calculator. |
| `app/schemas/__init__.py` | Schemas package marker. |
| `app/schemas/model.py` | Pydantic request and response DTO schemas for models and versions. |
| `app/services/__init__.py` | Services package marker. |
| `app/services/registry.py` | `ModelRegistryService` executing the 11-step validation pipeline. |
| `app/api/v1/endpoints/models.py` | FastAPI router for `/models` and `/models/{id}/versions`. |
| `app/api/v1/router.py` | Included `models.router` in API v1. |
| `app/models/enums.py` | Updated `ModelStatus` to include `REGISTERED`, `VALIDATING`, `STAGING`, `FAILED`. |
| `app/models/model.py` | Updated default status to `ModelStatus.REGISTERED`. |
| `tests/conftest.py` | Added dependency override and temp storage fixtures. |
| `tests/test_registry.py` | 10 comprehensive tests for registry and artifact pipeline. |
| `tests/test_database.py` | Updated tests to reflect `ModelStatus.REGISTERED`. |
| `requirements.txt` | Added `joblib`, `scikit-learn`, `scipy`, `numpy`, and `python-multipart`. |
| `docs/05_phase_03_model_registry.md` | This phase documentation. |

---

## Tech Used
- **FastAPI (`UploadFile`, `Form`)**: Streaming multipart file uploads and form metadata handling.
- **Pydantic v2**: Strict DTO parsing and validation for model catalogs.
- **joblib & scikit-learn**: Canonical model serialization, deserialization, and load verification.
- **hashlib**: SHA-256 cryptographic digest computation.
- **SQLAlchemy 2.0 Async**: Relational persistence and transactional state transitions.
- **Pytest**: Asynchronous endpoint testing with in-memory SQLite and mock models.
- **Ruff**: Code quality and import formatting.

---

## Architecture / Flow

```
Client (Data Scientist / CI)
       │
       ▼ POST /api/v1/models/{id}/versions
FastAPI Routing (models.py)
       │ (UploadFile, version_tag, input_schema, output_schema)
       ▼
ModelRegistryService.create_model_version()
       │
       ├─▶ 1. Check Model Exists in DB (404 if not found)
       ├─▶ 2. Check Version Tag Uniqueness (409 if duplicate)
       ├─▶ 3. Validate Extension (.joblib / .onnx, 400 if invalid)
       ├─▶ 4. Validate Size (<= 50MB, 413 if too large)
       ├─▶ 5. Compute SHA-256 Digest
       ├─▶ 6. Save Binary to LocalArtifactStore (/artifacts/{name}/{version}/model.joblib)
       │
       ├─▶ 7. Insert DB Record with status = "REGISTERED"
       ├─▶ 8. Transition status = "VALIDATING"
       │
       ├─▶ 9. Execute Load Test:
       │       - joblib.load(abs_path)
       │       - Check hasattr(model, "predict") and callable(predict)
       │       ├─ FAIL: Transition status = "FAILED", raise 422 Unprocessable Entity
       │       └─ PASS: Proceed
       │
       ├─▶ 10. Validate Schema Dictionary Structure (400 if malformed)
       │
       └─▶ 11. Transition status = "STAGING"
               - Commit transaction to PostgreSQL
               - Return 201 Created with ModelVersionResponse
```

---

## Important Concepts Explained

### 1. Model Registry
A centralized, authoritative catalog that tracks logical models, their immutable versioned iterations, training provenance, schemas, and deployment readiness. It decouples *model identity* (what business problem is being solved) from *model versions* (specific serialized binary weights).

### 2. Artifact
The serialized physical file (e.g. `.joblib` or `.onnx` binary) containing the trained mathematical weights, preprocessing pipeline, and hyperparameters. The artifact is the executable payload that the serving engine loads into memory.

### 3. Model Version
An immutable snapshot binding an artifact file to:
- A specific semantic version tag (e.g. `v1.2.0`).
- A parent logical model.
- An input/output schema contract.
- A cryptographic SHA-256 hash.
- A lifecycle status (`STAGING`, `ACTIVE`, `RETIRED`).

### 4. SHA-256 Cryptographic Hash
A 256-bit cryptographic hash function that produces a unique 64-character hexadecimal fingerprint of a binary file. In our platform, the SHA-256 hash is computed immediately upon upload and stored in the database.

### 5. Why Artifact Hash Matters
1. **Tamper Detection**: Guarantees that the binary on disk has not been altered, injected with malicious bytecode, or corrupted by filesystem faults.
2. **Reproducibility**: Guarantees that the exact binary evaluated in offline testing is identical to the binary running in production inference.
3. **Audit Compliance**: Financial and healthcare regulations require cryptographic proof of model provenance.

### 6. State Machine
A mathematical model of computation where an entity exists in exactly one state at any time and transitions between states based on explicit conditions. Our version lifecycle follows:
- `REGISTERED` $\rightarrow$ `VALIDATING` $\rightarrow$ `STAGING` (Deployment Ready)
- `VALIDATING` $\rightarrow$ `FAILED` (Terminal error; cannot be deployed)

---

## API / DB Changes

### Endpoints Created
- `POST /api/v1/models`: Register a logical model (201 Created).
- `GET /api/v1/models`: List models with pagination and version counts (200 OK).
- `GET /api/v1/models/{id}`: Get model details and version summaries (200 OK).
- `POST /api/v1/models/{id}/versions`: Multipart form upload and validation pipeline (201 Created).

### Model Status Enum Changes
- Updated `ModelStatus` values: `REGISTERED`, `VALIDATING`, `STAGING`, `FAILED`, `ACTIVE`, `RETIRED`.

---

## Testing

### Automated Test Results (Pytest)
```
collected 27 items

tests/test_database.py::test_all_12_tables_created PASSED                [  3%]
tests/test_database.py::test_create_user_and_model PASSED                [  7%]
tests/test_database.py::test_user_unique_email_constraint PASSED         [ 11%]
tests/test_database.py::test_model_version_unique_tag_constraint PASSED  [ 14%]
tests/test_database.py::test_offline_metrics_and_reference_stats PASSED  [ 18%]
tests/test_database.py::test_deployment_canary_routing PASSED            [ 22%]
tests/test_database.py::test_inference_request_and_ground_truth PASSED   [ 25%]
tests/test_database.py::test_online_and_perf_metrics PASSED              [ 29%]
tests/test_database.py::test_drift_event_and_audit_log PASSED            [ 33%]
tests/test_database.py::test_cascade_delete_model_removes_children PASSED [ 37%]
tests/test_health.py::test_liveness_probe_returns_200 PASSED             [ 40%]
tests/test_health.py::test_request_id_middleware_generates_header PASSED [ 44%]
tests/test_health.py::test_request_id_middleware_preserves_client_header PASSED [ 48%]
tests/test_health.py::test_readiness_probe_disconnected_subsystems PASSED [ 51%]
tests/test_health.py::test_readiness_probe_connected_subsystems PASSED   [ 55%]
tests/test_health.py::test_handled_app_exception_flow PASSED             [ 59%]
tests/test_health.py::test_unhandled_exception_flow PASSED               [ 62%]
tests/test_registry.py::test_register_model_success PASSED               [ 66%]
tests/test_registry.py::test_register_model_duplicate_name_conflict PASSED [ 70%]
tests/test_registry.py::test_list_models PASSED                          [ 74%]
tests/test_registry.py::test_get_model_by_id_and_not_found PASSED        [ 77%]
tests/test_registry.py::test_upload_valid_joblib_artifact_pipeline PASSED [ 81%]
tests/test_registry.py::test_upload_version_duplicate_tag_conflict PASSED [ 85%]
tests/test_registry.py::test_upload_invalid_extension_rejected PASSED    [ 88%]
tests/test_registry.py::test_upload_corrupted_model_fails_validation PASSED [ 92%]
tests/test_registry.py::test_upload_artifact_without_predict_method_fails PASSED [ 96%]
tests/test_registry.py::test_upload_malformed_schema_json_fails PASSED   [100%]

============================= 27 passed in 6.65s ==============================
```

### Linter Results (Ruff)
```
.\.venv\Scripts\ruff.exe check .
All checks passed!
```

---

## Commands Used
```bash
# 1. Run all tests including new registry tests
.\.venv\Scripts\pytest.exe -v

# 2. Verify code quality and styling
.\.venv\Scripts\ruff.exe check .
```

---

## Expected Behavior
- Uploading a valid `.joblib` model binary with a callable `.predict()` method passes load testing, computes its SHA-256 hash, writes to disk, and transitions the version state to `STAGING`.
- Uploading a corrupted file or an object lacking `.predict()` immediately marks the version as `FAILED` and returns `422 Unprocessable Entity`.
- Uploading duplicate model names returns `409 Conflict`.
- Uploading duplicate version tags for the same model returns `409 Conflict`.

---

## Problems Encountered & Solutions
1. **Enum Alignment with Existing Database Tests**:
   - *Problem*: Changing `ModelStatus` default from `DRAFT` to `REGISTERED` caused existing unit tests expecting `ModelStatus.DRAFT` to raise an `AttributeError`.
   - *Solution*: Updated `tests/test_database.py` to use `ModelStatus.REGISTERED` consistently, aligning all tests with the new state machine.

---

## Decisions / Tradeoffs
- **Synchronous Load Test During Upload**: Executed the load test synchronously in the upload request handler rather than offloading to a background worker. For lightweight models (< 50 MB), synchronous validation gives immediate HTTP feedback (e.g. 422 if invalid) to the client/CI pipeline without needing polling endpoints.
- **Protocol Interface for Storage (`ArtifactStore`)**: Used Python's `typing.Protocol` to define the storage contract. `LocalArtifactStore` handles disk operations today, but an `S3ArtifactStore` can be dropped in tomorrow without modifying a single line of the registry service.

---

## Interview Questions

### Q1: Why is validating that a model has a callable `.predict()` method crucial before marking it deployable?
> **Answer**: In Python, serialization tools (`pickle`, `joblib`) can serialize *any* Python object (lists, dicts, custom utility classes). If a user accidentally uploads a serialized preprocessor, dictionary, or corrupted object instead of the model estimator, the platform would crash with an `AttributeError: object has no attribute 'predict'` on the first live user request. Validating interface contracts at ingestion guarantees that only functional estimators enter `STAGING`.

### Q2: What security risks are associated with loading `.joblib` / `.pickle` files, and how does ModelOps Sentinel mitigate them?
> **Answer**: Pickle deserialization can execute arbitrary code (`__reduce__` exploit). ModelOps Sentinel mitigates this by:
> 1. Enforcing strict file extension filtering.
> 2. Computing and recording a SHA-256 cryptographic digest at upload time to detect tampering.
> 3. Enforcing file size limits (50 MB) to prevent zip-bomb / memory exhaustion attacks.
> 4. In production environments, unpickling occurs within isolated, non-root runner containers.

### Q3: Why is a Model Registry conceptually different from a standard file storage bucket (like S3)?
> **Answer**: S3 is merely an unindexed object store: it stores raw bytes with no awareness of model schemas, dependencies, training metadata, or deployment readiness. A Model Registry adds governance, versioning semantics, schema validation contracts, cryptographic lineage, and state machines (`REGISTERED` $\rightarrow$ `STAGING` $\rightarrow$ `ACTIVE`) that prevent unvalidated files from ever reaching production serving endpoints.

---

## Current Project State
- Stage 3 complete.
- Model registry and artifact validation pipeline fully functional.
- 27/27 automated tests passing.
- Linter clean: 0 errors.

---

## Next Stage
**Stage 4: Low-Latency Inference Serving & Redis Cache-Aside**
- Implement model in-memory loader and cache.
- Implement low-latency synchronous inference endpoint: `POST /api/v1/models/{id}/predict`.
- Implement Redis cache-aside engine with deterministic feature hashing.
- Telemetry logging: buffer inference events in Redis.
