# Phase 02 — PostgreSQL Persistence & Database Foundation

## Objective
Implement enterprise-grade persistence for **ModelOps Sentinel** using SQLAlchemy 2.0 Async, asyncpg, and Alembic. Establish the locked relational schema across all 12 platform entities with strict foreign keys, unique constraints, performance indexes, and transactional session management.

---

## Why This Stage Exists
In a production ML platform, the database is the immutable, auditable source of truth. Without an explicit relational schema:
1. **Model Provenance is Lost**: You cannot trace which exact dataset, code version, or engineer created a production artifact.
2. **Release Disasters Occur**: Without relational integrity between deployments, canary versions, and primary versions, rolling back or traffic routing creates orphaned, broken endpoints.
3. **Audit Invalidation**: Drift events, performance anomalies, and user actions must be permanently retained for regulatory and operational compliance.
4. **Race Conditions & Concurrency**: High-throughput inference logging and concurrent administrative actions demand ACID transactions and connection pooling.

---

## What Was Implemented
- **Base Infrastructure (`app/db/`)**:
  - `Base`: SQLAlchemy 2.0 `DeclarativeBase`.
  - `TimestampMixin`: Automatically injects timezone-aware `created_at` and `updated_at` (UTC) on entities.
  - `GUID`: Universal UUID type compiling to native PostgreSQL `UUID` or `CHAR(36)` on SQLite for fast isolated testing.
  - `JSONBCompat`: Universal JSON type compiling to PostgreSQL `JSONB` or standard `JSON` on SQLite.
  - `session.py`: Asynchronous engine (`create_async_engine`) with connection pooling (`pool_size`, `max_overflow`, `pool_pre_ping=True`) and scoped session dependency (`get_db_session`).
- **All 12 Platform Entities (`app/models/`)**:
  1. `User` (`users`): RBAC accounts (`ADMIN`, `OPERATOR`, `VIEWER`).
  2. `Model` (`models`): Logical ML entities with task types.
  3. `ModelVersion` (`model_versions`): Immutable versioned artifacts with SHA-256 hashes, input/output schemas, and lifecycle states (`DRAFT`, `STAGING`, `ACTIVE`, `RETIRED`).
  4. `OfflineMetrics` (`offline_metrics`): Pre-promotion test set benchmarks (accuracy, precision, recall, f1, roc_auc, custom).
  5. `ReferenceStats` (`reference_stats`): Baseline feature distributions and class frequencies used for drift detection comparison.
  6. `Deployment` (`deployments`): Traffic routing configurations supporting `DIRECT`, `CANARY`, and `SHADOW` strategies.
  7. `InferenceRequest` (`inference_requests`): Real-time telemetry records linking predictions, latencies, and correlation IDs.
  8. `GroundTruth` (`ground_truth`): Delayed real-world labels joined back to inference requests for online accuracy evaluation.
  9. `OnlineMetrics` (`online_metrics`): Sliding-window evaluation metrics derived from matched ground-truth.
  10. `PerfMetrics` (`perf_metrics`): Operational serving performance (p50, p95, p99 latency, cache hits, errors).
  11. `DriftEvent` (`drift_events`): Statistical covariate shift reports generated out-of-band via PSI or KS-tests.
  12. `AuditLog` (`audit_logs`): Immutable governance logs tracking who modified what, with before/after state payloads.
- **Alembic Migration System (`migrations/`)**:
  - `alembic.ini` and `migrations/env.py` configured for asynchronous execution with `Base.metadata`.
  - Initial migration revision (`0001_initial_schema.py`) containing complete transactional DDL for all 12 tables, indexes, constraints, and reverse downgrade script.
- **Automated Test Suite**:
  - 10 database integration tests covering table creation, relationships, foreign key cascade deletions, and unique constraint violations (17 total tests passing).

---

## Files Created/Modified

| File | Purpose |
| :--- | :--- |
| `app/db/__init__.py` | Database infrastructure package marker. |
| `app/db/base.py` | `DeclarativeBase`, `GUID`, `JSONBCompat`, and `TimestampMixin`. |
| `app/db/session.py` | Async engine, sessionmaker, and `get_db_session` dependency. |
| `app/models/__init__.py` | Re-exports all 12 models and enums for `Base.metadata`. |
| `app/models/enums.py` | Domain enums (`UserRole`, `ModelStatus`, `ArtifactFormat`, `DeploymentStrategy`, etc.). |
| `app/models/user.py` | `User` entity. |
| `app/models/model.py` | `Model` and `ModelVersion` entities. |
| `app/models/metrics.py` | `OfflineMetrics`, `ReferenceStats`, `OnlineMetrics`, `PerfMetrics`. |
| `app/models/deployment.py` | `Deployment` entity. |
| `app/models/inference.py` | `InferenceRequest` and `GroundTruth` entities. |
| `app/models/monitoring.py` | `DriftEvent` and `AuditLog` entities. |
| `alembic.ini` | Alembic migration configuration. |
| `migrations/env.py` | Async Alembic environment runner. |
| `migrations/script.py.mako` | Migration revision file template. |
| `migrations/versions/0001_initial_schema.py` | Initial migration script creating all 12 tables. |
| `tests/test_database.py` | 10 async database tests verifying CRUD, relations, cascades, and constraints. |
| `requirements.txt` | Added `sqlalchemy>=2.0.0`, `alembic>=1.13.0`, `aiosqlite>=0.20.0`. |
| `docs/04_phase_02_database.md` | This phase documentation. |

---

## Tech Used
- **SQLAlchemy 2.0 Async**: Declarative ORM using `Mapped` and `mapped_column` type annotations.
- **asyncpg**: High-performance PostgreSQL async driver.
- **Alembic**: Database migration tool configured for async connection transactions.
- **aiosqlite**: Embedded async SQLite driver used for lightning-fast, zero-dependency unit tests.
- **Pytest Asyncio**: Asynchronous test runner verifying relational models.
- **Ruff**: Enforces PEP 8 and deterministic import sorting.

---

## Architecture / Flow

```
FastAPI Request / Worker Coroutine
             │
             ▼
      get_db_session()
             │ (AsyncSession via AsyncSessionLocal)
             ▼
┌────────────────────────────────────────────────────────┐
│                   SQLAlchemy 2.0 ORM                   │
│                                                        │
│  User ──────1:N─────▶ Model ──────1:N──▶ ModelVersion │
│                         │                      │       │
│                         ▼                      ├──1:1──▶ OfflineMetrics
│                    Deployment ◀──1:N───────────┼──1:1──▶ ReferenceStats
│                         │                      ├──1:N──▶ OnlineMetrics
│                         ▼                      ├──1:N──▶ PerfMetrics
│                  InferenceRequest              ├──1:N──▶ DriftEvent
│                         │                      │       │
│                         └──1:1──▶ GroundTruth  ▼       │
│                                            AuditLog    │
└────────────────────────────────────────────────────────┘
             │
             ▼
   asyncpg (Binary Protocol)
             │
             ▼
   PostgreSQL 16 Engine
```

---

## Important Concepts Explained

### 1. ORM (Object-Relational Mapping)
An abstraction layer translating object-oriented code (Python classes, attributes) into relational database structures (tables, rows, columns, foreign keys). It eliminates raw SQL string concatenation, prevents SQL injection, and manages object identity.

### 2. Async DB Connection
Traditional database drivers (like `psycopg2`) block the thread while waiting for the database server to respond over the TCP socket. `asyncpg` yields control back to the Python asyncio event loop during network I/O, allowing a single worker process to handle thousands of concurrent queries without thread context-switching overhead.

### 3. Session (`AsyncSession`)
The unit-of-work container that tracks changes to objects (new, dirty, deleted) in memory. A session is **not** thread-safe or coroutine-shared; it must be scoped per request and disposed of at request termination.

### 4. Database Transaction
A group of database operations that execute under ACID guarantees: either all changes commit successfully, or all changes roll back completely on failure. In our platform, promotions and deployment updates are wrapped in `async with session.begin(): ...` to guarantee zero partial state corruption.

### 5. Migration (Alembic)
Version control for database schemas. Tracks incremental schema diffs in code (`upgrade()` / `downgrade()`), preventing schema drift across local, staging, and production environments.

### 6. Foreign Key & Cascade Deletes
A constraint enforcing referential integrity between tables. In our schema:
- Deleting a `Model` automatically cascades to delete all its `ModelVersion` records (`ondelete="CASCADE"`).
- Deleting an active `ModelVersion` referenced by a live `Deployment` is forbidden (`ondelete="RESTRICT"`), preventing accidental serving outages.

### 7. Database Index
A B-Tree auxiliary data structure that reduces query time from $O(N)$ (full table scan) to $O(\log N)$. Indexes are placed on foreign keys, version tags, query timestamps, and lookup fields like `request_id`.

### 8. JSONB
Decomposed binary JSON storage in PostgreSQL. Unlike rigid relational columns, `JSONB` stores arbitrary JSON (e.g., input features, metrics payloads, prediction outputs) while still allowing GIN indexing and fast key-value lookups.

### 9. Unique Constraint
Guarantees that no two rows share identical values across specified columns. Example: `uq_model_versions_model_tag` guarantees that for a given model, `version_tag` (e.g. `v1.0.0`) cannot be registered twice.

---

## API / DB Changes

### 12 Database Tables Created

| Table Name | Description | Key Columns & Constraints |
| :--- | :--- | :--- |
| `users` | Accounts & RBAC | `id` (PK), `email` (UQ, IX), `role`, `is_active` |
| `models` | Logical ML Model | `id` (PK), `name` (UQ, IX), `task_type`, `created_by_id` (FK) |
| `model_versions` | Versioned Artifacts | `id` (PK), `model_id` (FK), `version_tag` (IX), `status`, `input_schema` (JSONB), UQ(`model_id`, `version_tag`) |
| `offline_metrics` | Pre-deploy benchmarks | `id` (PK), `model_version_id` (FK, UQ), `accuracy`, `precision`, `f1_score`, `roc_auc` |
| `reference_stats` | Baseline drift data | `id` (PK), `model_version_id` (FK, UQ), `num_samples`, `feature_stats` (JSONB) |
| `deployments` | Traffic routing & Canary | `id` (PK), `model_id` (FK), `primary_version_id` (FK), `canary_version_id` (FK), `traffic_split` |
| `inference_requests` | Production telemetry | `id` (PK), `request_id` (IX), `model_version_id` (FK), `input_features` (JSONB), `prediction` (JSONB), `latency_ms` |
| `ground_truth` | Delayed target labels | `id` (PK), `inference_request_id` (FK, UQ), `actual_label` (JSONB), `matched_at` |
| `online_metrics` | Sliding accuracy window | `id` (PK), `model_version_id` (FK), `window_start`, `window_end`, UQ(`model_version_id`, `window_start`, `window_end`) |
| `perf_metrics` | Serving latencies | `id` (PK), `model_version_id` (FK), `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`, `error_count` |
| `drift_events` | Out-of-band drift alerts | `id` (PK), `model_version_id` (FK), `drift_detected` (IX), `metric_used`, `overall_drift_score`, `feature_drift_details` (JSONB) |
| `audit_logs` | Compliance log | `id` (PK), `user_id` (FK), `action` (IX), `entity_type` (IX), `payload_before` (JSONB), `payload_after` (JSONB) |

---

## Testing

### Automated Test Results (Pytest)
```
collected 17 items

tests/test_database.py::test_all_12_tables_created PASSED                [  5%]
tests/test_database.py::test_create_user_and_model PASSED                [ 11%]
tests/test_database.py::test_user_unique_email_constraint PASSED         [ 17%]
tests/test_database.py::test_model_version_unique_tag_constraint PASSED  [ 23%]
tests/test_database.py::test_offline_metrics_and_reference_stats PASSED  [ 29%]
tests/test_database.py::test_deployment_canary_routing PASSED           [ 35%]
tests/test_database.py::test_inference_request_and_ground_truth PASSED  [ 41%]
tests/test_database.py::test_online_and_perf_metrics PASSED             [ 47%]
tests/test_database.py::test_drift_event_and_audit_log PASSED           [ 52%]
tests/test_database.py::test_cascade_delete_model_removes_children PASSED [ 58%]
tests/test_health.py::test_liveness_probe_returns_200 PASSED             [ 64%]
tests/test_health.py::test_request_id_middleware_generates_header PASSED  [ 70%]
tests/test_health.py::test_request_id_middleware_preserves_client_header PASSED [ 76%]
tests/test_health.py::test_readiness_probe_disconnected_subsystems PASSED [ 82%]
tests/test_health.py::test_readiness_probe_connected_subsystems PASSED    [ 88%]
tests/test_health.py::test_handled_app_exception_flow PASSED             [ 94%]
tests/test_health.py::test_unhandled_exception_flow PASSED               [100%]

============================= 17 passed in 3.86s ==============================
```

### Alembic Migration Verification
1. **Offline SQL Upgrade (`alembic upgrade head --sql`)**: Verified generation of transactional PostgreSQL DDL creating all 12 tables, indexes, foreign keys, and version table.
2. **Offline SQL Downgrade (`alembic downgrade 0001_initial_schema:base --sql`)**: Verified generation of reverse drop table script in exact topological dependency order.

### Linter Results (Ruff)
```
.\.venv\Scripts\ruff.exe check .
All checks passed!
```

---

## Commands Used
```bash
# 1. Generate offline migration SQL
.\.venv\Scripts\alembic.exe upgrade head --sql

# 2. Test rollback SQL generation
.\.venv\Scripts\alembic.exe downgrade 0001_initial_schema:base --sql

# 3. Run complete test suite (unit, health, database)
.\.venv\Scripts\pytest.exe

# 4. Verify code formatting and linting
.\.venv\Scripts\ruff.exe check .
```

---

## Expected Behavior
- When migrations run on an empty database, all 12 tables and auxiliary indexes are created transactionally.
- Any attempt to insert a duplicate email or duplicate `(model_id, version_tag)` raises an `IntegrityError`.
- Relationships configure `lazy="selectin"` to prevent async `MissingGreenlet` exceptions during relationship traversal.
- Deleting a parent `Model` cleanly purges its versions and associated child metrics without leaving orphaned records.

---

## Problems Encountered & Solutions
1. **Async SQLAlchemy `MissingGreenlet` on Relationship Access**:
   - *Problem*: In async SQLAlchemy, accessing a relationship attribute (e.g. `fetched_version.offline_metrics`) synchronously raises `MissingGreenlet` because lazy-loading triggers implicit blocking I/O on the event loop.
   - *Solution*: Configured `lazy="selectin"` on 1-to-1 and parent-child relationships, instructing SQLAlchemy to eagerly load related records using an optimized `IN` query during the primary async fetch.

---

## Decisions / Tradeoffs
- **Universal `GUID` and `JSONBCompat` Custom Types**: Built custom type decorators that compile to PostgreSQL native `UUID` and `JSONB` in production, while falling back to `CHAR(36)` and `JSON` on SQLite. This allows developers to run the entire automated test suite locally in 3 seconds without requiring an external PostgreSQL daemon.
- **RESTRICT on Deployment Foreign Keys**: Used `ondelete="RESTRICT"` for `primary_version_id` and `canary_version_id` on the `deployments` table. If an operator attempts to delete a model version that is currently actively serving live production traffic, the database directly rejects the deletion at the constraint level, eliminating catastrophic accidental outages.

---

## Interview Questions

### Q1: What is the difference between `lazy="selectin"` and `lazy="joined"` in SQLAlchemy 2.0 async?
> **Answer**: `joined` executes a SQL `JOIN` in the original query. While efficient for 1-to-1 relationships, using `joined` on 1-to-many relationships causes result set multiplication (Cartesian product), leading to memory bloat and duplicate row parsing. `selectin` executes a second query using an `IN (<id_list>)` clause. In async code, `selectin` is preferred for both 1-to-1 and collections because it produces clean SQL, prevents Cartesian explosion, and works asynchronously without triggering `MissingGreenlet`.

### Q2: Why is `ON DELETE RESTRICT` critical for production deployment tables?
> **Answer**: In ML systems, a model version might be referenced by a deployment routing table currently receiving live client inference requests. If a database rule allows cascade deletion (`CASCADE`) or nullification (`SET NULL`), an administrative script deleting an old version could silently delete or break the live deployment, causing incoming HTTP traffic to crash with 500 errors. `RESTRICT` guarantees at the database engine level that an active version cannot be deleted until it has been explicitly decommissioned and removed from all deployment routing tables.

### Q3: What is the difference between `alembic upgrade head` and `alembic upgrade head --sql`?
> **Answer**: `alembic upgrade head` runs in "online" mode: it opens a live connection to the database and executes DDL statements directly. `alembic upgrade head --sql` runs in "offline" mode: it reads the migration scripts and outputs the raw, formatted SQL DDL statements (within a transaction block `BEGIN...COMMIT`) to standard output without touching a database. Offline mode is standard in enterprise CI/CD where database administrators (DBAs) or security gates review migration scripts before execution.

---

## Current Project State
- Stage 2 complete.
- PostgreSQL schema for all 12 tables established.
- Alembic async migration configuration and initial revision verified.
- 17/17 automated tests passing.
- Linter clean.

---

## Next Stage
**Stage 3: Authentication, RBAC & API Key Security**
- Implement JWT token generation and validation.
- Implement API Key generation, hashing (SHA-256 / bcrypt), and validation for machine-to-machine inference clients.
- Build RBAC authorization dependencies (`require_role(UserRole.ADMIN)`).
- Create auth endpoints: `/api/v1/auth/login`, `/api/v1/auth/api-keys`.
