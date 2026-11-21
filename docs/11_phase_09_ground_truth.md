# Stage 9: Delayed Ground-Truth Evaluation & Performance Tracking

## 1. Architectural Positioning & Philosophy

In **ModelOps Sentinel**, **Delayed Ground-Truth Evaluation** closes the observability loop between live production predictions and real-world business outcomes.

In production machine learning systems, serving a prediction takes milliseconds—but learning whether that prediction was actually *right* often takes hours, days, weeks, or even months. Stage 7 gave us real-time **online serving telemetry** (latency, throughput, error rates, cache hits), and Stage 8 gave us **statistical data drift monitoring** (early warning when input distributions shift). Stage 9 adds the final ground-truth verification pillar: matching delayed real-world outcome labels back to historical predictions via `request_id` and calculating true production predictive performance (`accuracy`, `precision`, `recall`, `F1`, and `confusion_matrix`) out-of-band in background workers.

### Lightweight ModelOps vs. Generic MLOps
Generic MLOps platforms frequently couple delayed evaluation to complex streaming join pipelines (Kafka + Flink + Feature Stores + external evaluation lakes) that introduce heavy operational overhead.

**ModelOps Sentinel** keeps the architecture lightweight, deterministic, and production-realistic:
1. **Correlation by `request_id`**: Every logged `InferenceRequest` carries a deterministic `request_id` and `model_version_id`.
2. **Simple, Strict Feedback Ingestion**: Downstream business systems submit resolved outcomes via `POST /api/v1/models/{id}/feedback` with `{ "request_id": "...", "label": ... }`. The endpoint verifies model ownership (preventing cross-model poisoning), enforces idempotency against duplicate labels (`409 Conflict`), and persists immutable records into `ground_truth`.
3. **Decoupled Worker Calculation**: A background worker task (`evaluate_ground_truth_task`) joins `inference_requests` with `ground_truth`, computes `accuracy`, `precision`, `recall`, `F1`, and the structured `confusion_matrix`, and persists snapshots into `perf_metrics`.
4. **Strict Three-Way Metric Separation**: Offline metrics, online serving metrics, and delayed ground-truth performance are stored, queried, and reasoned about independently—never mixed.

```
+---------------------------------------------------------------------------------------------------+
|                             STRICT THREE-WAY METRIC SEPARATION                                    |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  1. OFFLINE METRICS (Pre-Deployment, Static per ModelVersion)                                     |
|     Holdout / Test Set ------> Offline Evaluation ------> offline_metrics (PostgreSQL)            |
|                                                                                                   |
|  2. ONLINE METRICS (Real-Time Inference Telemetry, Zero Label Delay)                              |
|     Client -> POST /predict -> Fast InferenceEngine ----> Prometheus /metrics (Latency, RPS,      |
|                      |                                    Errors, Cache Hits) & online_metrics    |
|                      +------> Log InferenceRequest (request_id, prediction, timestamp)            |
|                                      |                                                            |
|  3. DELAYED GROUND-TRUTH PERFORMANCE | (Arrives minutes / hours / days / months later)            |
|                                      |                                                            |
|     Real-World Outcome Observed -----+--> POST /api/v1/models/{id}/feedback                       |
|     (chargeback, default, audit)          { "request_id": "...", "label": 1 }                     |
|                                                         |                                         |
|                                                         v                                         |
|                                                   ground_truth                                    |
|                                                         |                                         |
|                                                         v                                         |
|                                     Background Worker (evaluate_ground_truth_task)                |
|                                     Joins inference_requests <-> ground_truth                     |
|                                     Computes: accuracy, precision, recall, F1, confusion_matrix   |
|                                                         |                                         |
|                                                         v                                         |
|                                                   perf_metrics                                    |
|                                                         |                                         |
|                                                         v                                         |
|                                     GET /api/v1/models/{id}/performance                           |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Why Offline Metrics, Online Metrics, and Delayed Ground-Truth Performance Are Different

> [!IMPORTANT]
> **Offline metrics, online metrics, and delayed ground-truth performance measure fundamentally different properties on different timelines over different populations. Mixing them corrupts governance, breaks alerting, and misleads incident response.**

### 1. Offline Metrics (`offline_metrics`)
- **What they are**: Predictive quality metrics (`accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`) computed **before deployment** on a curated, static historical holdout/validation/test dataset.
- **When available**: Once at model registration/validation time, before a version is promoted to `STAGING` or `ACTIVE`.
- **What they tell you**: Whether the trained artifact learned the historical problem well enough to pass release gates relative to prior versions.
- **Blind spots**: Completely blind to production reality—train/serve feature skew, upstream data pipeline bugs, covariate drift, adversarial user behavior, and real-world concept drift.

### 2. Online Metrics (Prometheus `/metrics` & Real-Time Operational Telemetry)
- **What they are**: Immediate, serving-time operational and behavioral telemetry observed **at the instant of inference** without knowing the true outcome: HTTP request rate (RPS), error rate (4xx/5xx), end-to-end and model-only latency percentiles ($p50$, $p95$, $p99$), cache hit rate, async job throughput, and live prediction confidence/class distribution.
- **When available**: Immediately (milliseconds to seconds) via Prometheus scraping and inference telemetry.
- **What they tell you**: Whether the serving system is healthy, fast, available, and operating within SLA budgets right now.
- **Blind spots**: Completely blind to **predictive correctness**. A model can serve predictions in `1.5ms` with `0%` HTTP errors, `95%` cache hits, and `0.99` average confidence while predicting the wrong class on 40% of real transactions due to concept drift.

### 3. Delayed Ground-Truth Performance (`perf_metrics` via `ground_truth`)
- **What they are**: True production predictive quality (`accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`) computed strictly by joining historical logged predictions (`inference_requests`) with confirmed real-world outcomes (`ground_truth`).
- **When available**: Delayed—ranging from minutes (ad clicks, quick operator feedback) to hours/days (industrial anomaly root-cause investigation) to weeks/months (fraud chargebacks, loan defaults).
- **What they tell you**: Whether the model is actually making correct decisions in the real world after exposure to live traffic and distribution shifts.
- **Blind spots**:
  - **Label latency**: Cannot trigger sub-minute rollback during an immediate broken deploy (that is the job of online metrics and canary health checks).
  - **Partial coverage & selection bias**: Often only a subset of predictions ever receive ground-truth labels (tracked explicitly via `label_coverage_ratio`). For example, in credit lending, you only observe loan repayment outcomes for applicants who were *approved*, never for applicants who were *rejected*.

### Side-by-Side Comparison Matrix

| Dimension | Offline Metrics | Online Metrics | Delayed Ground-Truth Performance |
| :--- | :--- | :--- | :--- |
| **Storage / Channel** | `offline_metrics` table | Prometheus `/metrics` (TSDB) & `online_metrics` | `perf_metrics` table (from `ground_truth`) |
| **Lifecycle Stage** | Pre-deployment (Validation Gate) | Real-time Production Serving | Post-outcome Production Auditing |
| **Requires Ground Truth?** | Yes (historical test split) | **No** | Yes (delayed real-world labels) |
| **Time Availability** | Static (computed once per version) | Immediate (milliseconds / seconds) | Delayed (minutes to months) |
| **Core Signals** | Test Accuracy, F1, ROC-AUC | p50/p95/p99 latency, RPS, error rate, cache hits | Live Accuracy, Precision, Recall, F1, Confusion Matrix |
| **Primary Question Answered** | *"Is this artifact good enough to deploy?"* | *"Is the serving endpoint fast and healthy right now?"* | *"Were our past production predictions actually correct?"* |
| **Primary Blind Spot** | Production drift & train/serve skew | Silent accuracy collapse | Label delay & selection/censoring bias |

### Why Mixing Them Causes Production Failures
1. **Overwriting `offline_metrics` with production results destroys auditability**: You lose the immutable pre-deployment benchmark needed to measure how far production performance has degraded from test-set expectations (offline-to-online gap).
2. **Mixing online operational telemetry with delayed ground-truth metrics breaks alerting and rollbacks**: Operational alerts need sub-minute latency/error evaluation across 100% of live traffic, whereas delayed ground-truth evaluation operates over historical prediction windows (`window_start` to `window_end`) with partial label coverage. Combining them creates misleading timestamps (attributing a 30-day-old prediction's outcome to *today's* serving window) and false alarms.
3. **Treating delayed ground-truth metrics as real-time online accuracy hides censoring bias**: If only 15% of predictions have received feedback so far—and failures are reported faster than successes—displaying that number next to live RPS without separating `metric_category` and `label_coverage_ratio` leads engineers to draw false conclusions about current model health.

---

## 3. Core Concepts: Delayed Feedback & Classification Metrics

### Delayed Ground-Truth Matching
Every prediction served by `InferenceEngine` or `async_predict_task` writes an `InferenceRequest` containing:
- `request_id`: Unique correlation identifier returned to the caller.
- `model_version_id`: Exact immutable `ModelVersion` that served the prediction.
- `prediction`: Stored JSON payload `{"value": <pred>, "confidence": <prob>}`.
- `created_at`: Exact UTC timestamp when the prediction was made.

When the real-world outcome becomes known later, the calling service submits `request_id` and `label` to `POST /api/v1/models/{id}/feedback`. The platform joins the outcome back to the exact `InferenceRequest` and records `matched_at`, allowing calculation of `feedback_delay_seconds` and preventing duplicate label submissions via a unique constraint on `ground_truth.inference_request_id`.

### Classification Metrics Calculated by the Worker

For a window of matched `(y_true, y_pred)` pairs, the worker computes:

1. **Confusion Matrix**:
   - **Binary classification**: Explicit $2 \times 2$ matrix ordered as `[negative, positive]` (canonical `[0, 1]` for standard binary models):
     $$\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$
     with named counts `tp`, `tn`, `fp`, `fn` persisted in JSONB.
   - **Multiclass classification**: $K \times K$ matrix over sorted class labels with weighted averaging.
2. **Accuracy**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
3. **Precision** (Positive Predictive Value):
   $$\text{Precision} = \frac{TP}{TP + FP}$$
4. **Recall** (Sensitivity / True Positive Rate):
   $$\text{Recall} = \frac{TP}{TP + FN}$$
5. **F1-Score** (Harmonic Mean of Precision & Recall):
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

All division operations use `zero_division=0` protection so skewed windows (e.g., zero predicted positives during quiet traffic periods) produce deterministic `0.0` values instead of crashing workers with `ZeroDivisionError` or storing `NaN`.

### How Stage 8 (Drift) and Stage 9 (Ground Truth) Work Together
- **Stage 8 (Data Drift)** detects shifts in $P(X)$ **before** ground-truth labels arrive (early warning).
- **Stage 9 (Delayed Ground-Truth)** measures true shifts in predictive performance after $Y$ arrives (definitive verification of concept drift or real model degradation).

---

## 4. Data Model & Storage

### `ground_truth` Table
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | PK | Unique feedback record identifier. |
| `inference_request_id` | `UUID` | FK (`inference_requests.id`, CASCADE), UNIQUE, INDEX | Strict 1-to-1 link to the original served prediction. |
| `actual_label` | `JSONB` | NOT NULL | Real-world outcome stored as `{"value": <label>}`. |
| `matched_at` | `DATETIME(TZ)` | NOT NULL | UTC timestamp when ground truth arrived. |
| `created_at` | `DATETIME(TZ)` | NOT NULL | Record creation timestamp. |

### `perf_metrics` Table (Delayed Ground-Truth Performance)
| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | `UUID` | PK | Unique evaluation snapshot identifier. |
| `model_version_id` | `UUID` | FK (`model_versions.id`, CASCADE), INDEX | Evaluated `ModelVersion`. |
| `window_start` | `DATETIME(TZ)` | NOT NULL, INDEX | Earliest `InferenceRequest.created_at` in evaluated cohort. |
| `window_end` | `DATETIME(TZ)` | NOT NULL, INDEX | Latest `InferenceRequest.created_at` in evaluated cohort. |
| `sample_count` | `INTEGER` | NOT NULL | Number of matched `(prediction, ground_truth)` pairs. |
| `accuracy` | `FLOAT` | NULLABLE | Computed classification accuracy. |
| `precision` | `FLOAT` | NULLABLE | Computed precision (binary or weighted). |
| `recall` | `FLOAT` | NULLABLE | Computed recall (binary or weighted). |
| `f1_score` | `FLOAT` | NULLABLE | Computed F1-score (binary or weighted). |
| `confusion_matrix` | `JSONB` | NULLABLE | Structured `labels`, 2D `matrix`, and `tp`/`tn`/`fp`/`fn`. |
| `evaluated_at` | `DATETIME(TZ)` | NOT NULL, INDEX | UTC timestamp when the worker calculated this snapshot. |

> [!NOTE]
> Alembic migration [`migrations/versions/0002_add_ground_truth_perf_metrics.py`](file:///d:/ModelOpsSentinel/migrations/versions/0002_add_ground_truth_perf_metrics.py) adds the delayed ground-truth columns to `perf_metrics` and relaxes legacy Stage 2 operational latency columns to nullable so Stage 9 worker writes strictly contain delayed ground-truth metrics with zero operational mixing.

---

## 5. API Specification

### 1. `POST /api/v1/models/{id}/feedback`
Submits a delayed real-world outcome label for a prior prediction.

#### Request
```http
POST /api/v1/models/33210f41-e8dd-4323-899a-e1a054164862/feedback HTTP/1.1
Host: localhost:8000
Authorization: Bearer <JWT_ACCESS_TOKEN>
Content-Type: application/json

{
  "request_id": "req-delayed-001",
  "label": 1
}
```

#### Response (`HTTP 201 Created`)
```json
{
  "id": "8e44a3c7-1e25-4452-b2f0-9a7e4a472901",
  "model_id": "33210f41-e8dd-4323-899a-e1a054164862",
  "model_version_id": "9f10b7d2-5a38-4f6a-8b9e-2d1c3e4f5a6b",
  "version_tag": "v1.0.0",
  "inference_request_id": "a1b2c3d4-0000-4000-8000-000000000001",
  "request_id": "req-delayed-001",
  "predicted_label": 1,
  "actual_label": 1,
  "is_correct": true,
  "feedback_delay_seconds": 7204.512,
  "matched_at": "2026-11-18T13:20:00.123456Z"
}
```

#### Error Semantics
- **`404 Not Found`**: Model `{id}` does not exist, `request_id` does not exist, or `request_id` belongs to a different model (cross-model isolation).
- **`409 Conflict` (`DUPLICATE_FEEDBACK`)**: Ground-truth feedback has already been recorded for this `request_id`.
- **`422 Unprocessable Entity`**: Missing or blank `request_id` or missing `label`.

---

### 2. `POST /api/v1/models/{id}/performance/evaluate`
Triggers delayed ground-truth evaluation immediately (`async_mode=false`, default) or enqueues `evaluate_ground_truth_task` onto the background worker (`async_mode=true`).

#### Request (Async Worker Mode)
```http
POST /api/v1/models/33210f41-e8dd-4323-899a-e1a054164862/performance/evaluate?async_mode=true HTTP/1.1
Host: localhost:8000
Authorization: Bearer <JWT_ACCESS_TOKEN>
```

#### Response (`HTTP 200 OK`)
```json
{
  "metric_category": "delayed_ground_truth_performance",
  "model_id": "33210f41-e8dd-4323-899a-e1a054164862",
  "model_name": "credit-default-gt",
  "version_tag": "v1.0.0",
  "job_id": "f4c9a0e1b2d34567890abcdef1234567",
  "status": "QUEUED"
}
```

---

### 3. `GET /api/v1/models/{id}/performance`
Retrieves delayed ground-truth performance snapshots from `perf_metrics` along with label coverage ratio.

#### Response (`HTTP 200 OK`)
```json
{
  "metric_category": "delayed_ground_truth_performance",
  "model_id": "33210f41-e8dd-4323-899a-e1a054164862",
  "model_name": "credit-default-gt",
  "version_tag": "v1.0.0",
  "total_inference_requests": 10,
  "matched_ground_truth_count": 8,
  "label_coverage_ratio": 0.8,
  "latest_metrics": {
    "id": "b72e1100-7d6c-414b-9e00-112233445566",
    "model_version_id": "9f10b7d2-5a38-4f6a-8b9e-2d1c3e4f5a6b",
    "window_start": "2026-11-15T13:20:00Z",
    "window_end": "2026-11-15T14:30:00Z",
    "sample_count": 8,
    "accuracy": 0.75,
    "precision": 0.8,
    "recall": 0.8,
    "f1_score": 0.8,
    "confusion_matrix": {
      "labels": [0, 1],
      "matrix": [[2, 1], [1, 4]],
      "tp": 4,
      "tn": 2,
      "fp": 1,
      "fn": 1,
      "averaging": "binary"
    },
    "evaluated_at": "2026-11-18T13:22:10Z"
  },
  "history": [ ... ]
}
```

---

## 6. Files Created / Modified

| File | Purpose |
| :--- | :--- |
| `app/models/metrics.py` | Extended `PerfMetrics` (`perf_metrics`) with delayed ground-truth columns (`sample_count`, `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `evaluated_at`) while preserving strict separation from `offline_metrics` and `online_metrics`. |
| `migrations/versions/0002_add_ground_truth_perf_metrics.py` | Reversible Alembic migration adding delayed ground-truth columns to `perf_metrics`. |
| `app/schemas/ground_truth.py` | Pydantic request/response contracts (`FeedbackCreateRequest`, `FeedbackResponse`, `ConfusionMatrixSchema`, `PerfMetricItem`, `ModelPerformanceResponse`). |
| `app/services/ground_truth.py` | Label extraction/alignment, classification metric calculation, `GroundTruthService` for feedback ingestion, delayed evaluation, and performance retrieval. |
| `app/worker/tasks.py` | Added `evaluate_ground_truth_task` background worker task and registered it in `TASK_REGISTRY`. |
| `app/worker/settings.py` | Registered `evaluate_drift_task` and `evaluate_ground_truth_task` in `WorkerSettings.functions`. |
| `app/api/v1/endpoints/models.py` | Added `POST /api/v1/models/{id}/feedback`, `GET /api/v1/models/{id}/performance`, and `POST /api/v1/models/{id}/performance/evaluate`. |
| `tests/test_ground_truth.py` | Automated test suite verifying metric math, delayed feedback API, duplicate/cross-model guards, worker calculation, and strict metric separation. |
| `docs/11_phase_09_ground_truth.md` | This Stage 9 documentation. |

---

## 7. Verification & Test Suite

Run Stage 9 tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ground_truth.py -v
```

Run full repository test suite & linter:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

| Test Case | Description | Status |
| :--- | :--- | :--- |
| `test_compute_binary_classification_metrics_exact` | Exact binary accuracy, precision, recall, F1, and TP/TN/FP/FN confusion matrix | **PASSED** |
| `test_compute_binary_metrics_zero_division_safe` | Safe zero-division handling when no positives are predicted | **PASSED** |
| `test_compute_multiclass_classification_metrics` | 3-class weighted precision/recall/F1 and $3 \times 3$ confusion matrix | **PASSED** |
| `test_compute_metrics_string_and_bool_label_alignment` | String labels and bool/int alignment normalization | **PASSED** |
| `test_submit_delayed_feedback_stores_ground_truth` | `POST /api/v1/models/{id}/feedback` persists `ground_truth` and computes delay | **PASSED** |
| `test_submit_duplicate_feedback_returns_409` | Rejects duplicate feedback for the same `request_id` with `409 DUPLICATE_FEEDBACK` | **PASSED** |
| `test_submit_feedback_not_found_and_cross_model_isolation` | Rejects unknown model, unknown `request_id`, and cross-model feedback with `404` | **PASSED** |
| `test_worker_evaluate_ground_truth_and_get_performance_api` | End-to-end delayed feedback -> worker `evaluate_ground_truth_task` -> `perf_metrics` -> `GET /performance` | **PASSED** |
| `test_worker_no_ground_truth_skips_perf_metrics_write` | Skips persisting bogus rows when no matched ground truth exists | **PASSED** |
| `test_trigger_performance_evaluate_sync_and_async_modes` | Verifies both immediate and background queue evaluation via API | **PASSED** |
| `test_strict_separation_of_offline_online_and_delayed_metrics` | Confirms delayed evaluation writes strictly to `perf_metrics` without mutating `offline_metrics`, `online_metrics`, or operational latency fields | **PASSED** |
