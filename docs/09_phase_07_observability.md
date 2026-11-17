# Stage 7: Telemetry, Prometheus Metrics & Grafana Observability

## 1. Architectural Positioning & Philosophy

In **ModelOps Sentinel**, observability is a first-class operational pillar designed to provide deep runtime transparency into machine learning model serving, API reliability, cache efficiency, and background asynchronous jobs.

### The Critical Architectural Separation: Ephemeral Telemetry vs. Database History
> [!IMPORTANT]
> **Prometheus metrics are NEVER stored in relational application databases (PostgreSQL or SQLite) as application history.**

- **Application Relational Database (PostgreSQL / SQLite)**:
  - Stores **durable business state**, **audit logs**, **user credentials**, **model metadata**, and **individual inference transaction logs** with high relational fidelity and transactional ACID guarantees.
  - Relational databases are poorly suited for high-frequency, continuous metric aggregation (thousands of samples per second across percentiles), which leads to lock contention, database bloat, and table write bottlenecks.
- **Prometheus Time-Series Engine (In-Memory Exposition + TSDB)**:
  - Metrics inside the FastAPI application process are stored in ultra-lightweight, in-process, lock-free **atomic memory buffers**.
  - Prometheus periodically queries `GET /metrics` via a **pull model** (scraping) and stores the compressed, timestamped values in its dedicated Time-Series Database (TSDB).
  - This guarantees that application telemetry adds virtually zero CPU overhead, never blocks API endpoints, and cannot cause relational database lock contention.

```
                           +------------------------------------------+
                           |           ModelOps Sentinel              |
                           |                                          |
Client Request ----------> |  1. FastAPI RequestContextMiddleware     |
                           |     -> Atomic in-memory metric increments|
                           |                                          |
                           |  2. InferenceEngine                      |
                           |     -> Calculates inference latency      |
                           |     -> Records cache hit/miss            |
                           |                                          |
                           |  3. Background Async Worker              |
                           |     -> Enqueue/Completion counters       |
                           +------------------------------------------+
                                               |
                                    GET /metrics (Exposition)
                                               |
                                               v
                                   +-----------------------+
                                   |      Prometheus       |
                                   |  (Scrapes every 5s)   |
                                   |   Time-Series DB      |
                                   +-----------------------+
                                               |
                                            PromQL
                                               |
                                               v
                                   +-----------------------+
                                   |        Grafana        |
                                   | Operational Dashboard |
                                   +-----------------------+
```

---

## 2. Core Concepts Explained

### 1. Metric
A **metric** is a numeric, timestamped quantitative measurement representing the state, rate, or behavior of a software system at a specific instant in time. Unlike logs (which record discrete events with text context) and traces (which follow a single transaction across boundaries), metrics are optimized for **real-time statistical aggregation, alerting, and trend analysis**.

### 2. Counter
A **counter** is a cumulative metric that represents a single monotonically increasing value. It can only increase or be reset to zero upon application restart (process reboot). Counters never decrease.
- **Why it matters**: Counters are immune to sampling interval artifacts. By computing the rate of increase over time using `rate()` or `irate()`, operators determine exact per-second throughput, request rates, or error frequencies regardless of how long the application has been running.
- **ModelOps Sentinel Counters**:
  - `modelops_http_requests_total`: Total HTTP calls served.
  - `modelops_http_errors_total`: Total HTTP responses with status code $\ge 400$.
  - `modelops_model_inferences_total`: Total predictions served per model and version.
  - `modelops_inference_cache_requests_total`: Cache lookups partitioned by `hit` or `miss`.
  - `modelops_async_jobs_total`: Background jobs partitioned by state (`enqueued`, `completed`, `failed`).

### 3. Histogram
A **histogram** samples observations (typically latencies, durations, or payload sizes) and counts them in configurable bucket boundaries. A Prometheus histogram generates three distinct underlying metrics:
1. `<metric_name>_bucket{le="<boundary>"}`: The cumulative count of observations whose value is less than or equal to the bucket upper bound `le`.
2. `<metric_name>_count`: The total number of observations recorded.
3. `<metric_name>_sum`: The arithmetic sum of all observed values.

- **Why Histograms over Averages for ML Serving**:
  Averages hide latency spikes. If 99 predictions take 5ms and 1 takes 5,000ms, the average is 55ms, masking catastrophic SLA violations. Histograms enable Prometheus to calculate precise percentiles ($p50$, $p95$, $p99$) using the mathematical function `histogram_quantile(φ, sum(rate(..._bucket[1m])) by (le))`.
- **ModelOps Sentinel Histograms**:
  - `modelops_http_request_duration_seconds`: Overall HTTP latency (Buckets: 5ms to 10s).
  - `modelops_inference_duration_seconds`: Pure model inference calculation latency (Buckets: 1ms to 2.5s).

### 4. Labels
**Labels** are multi-dimensional key-value pairs attached to a metric (e.g., `model="iris_classifier"`, `version="v1.0.0"`, `status_code="200"`). They allow fine-grained filtering and aggregation across arbitrary dimensions using PromQL (e.g. `by (model, version)`).
- **The Golden Rule: Beware High Cardinality!**
  Each unique permutation of label key-value pairs creates a completely new time series in Prometheus memory. Using dynamic values (like UUIDs, timestamps, or arbitrary user inputs) as label values leads to exponential memory growth and server crashes.
- **ModelOps Sentinel Cardinality Safeguard**:
  Dynamic route variables (e.g. `/api/v1/models/a1b2c3d4-e5f6.../predict`) are automatically normalized to route templates (`/api/v1/models/{id}/predict`) via `normalize_endpoint()` before being assigned to metric labels.

### 5. Prometheus Scraping
Prometheus uses a **pull-based** telemetry model:
- The monitored application exposes an HTTP endpoint (`GET /metrics`) returning plaintext in the Prometheus exposition format.
- Prometheus acts as an active scraper: on a configured schedule (e.g., every 5 seconds), it sends an HTTP GET to the target, consumes the payload, timestamps the values, and appends them to its TSDB.
- **Advantages**:
  - The application is decoupled from the monitoring stack; if Prometheus is offline, the application continues serving traffic unimpeded.
  - Simplifies service discovery, eliminates push storm thundering herd problems, and inherently detects application crashes (if scrape fails, target is marked DOWN).

### 6. Grafana Visualization
**Grafana** is an operational visualization platform that queries time-series databases (like Prometheus) using PromQL and translates raw mathematical vectors into intuitive real-time dashboards, gauges, graphs, and alert triggers.

---

## 3. Metrics Implementation Details

### Metrics Catalog

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `modelops_http_requests_total` | **Counter** | `method`, `endpoint`, `status_code` | Total count of HTTP requests processed by the platform. |
| `modelops_http_errors_total` | **Counter** | `method`, `endpoint`, `status_code` | Total count of HTTP error responses (HTTP status $\ge 400$). |
| `modelops_http_request_duration_seconds` | **Histogram** | `method`, `endpoint` | End-to-end HTTP request latency in seconds. |
| `modelops_inference_duration_seconds` | **Histogram** | `model`, `version` | Pure scikit-learn model inference execution duration in seconds. |
| `modelops_model_inferences_total` | **Counter** | `model`, `version`, `status` | Predictions served per model and version tag. |
| `modelops_async_jobs_total` | **Counter** | `job_type`, `status` | Asynchronous background jobs (`enqueued`, `completed`, `failed`). |
| `modelops_inference_cache_requests_total` | **Counter** | `model`, `version`, `result` | Cache performance lookups (`hit` vs `miss`). |

### Exposition Endpoint
- **URL**: `GET /metrics`
- **Content-Type**: `text/plain; version=0.0.4; charset=utf-8`
- **Output Sample**:
```text
# HELP modelops_http_requests_total Total count of HTTP requests processed by the platform.
# TYPE modelops_http_requests_total counter
modelops_http_requests_total{endpoint="/api/v1/health",method="GET",status_code="200"} 12.0
modelops_http_requests_total{endpoint="/api/v1/models/{id}/predict",method="POST",status_code="200"} 45.0

# HELP modelops_inference_cache_requests_total Inference result cache lookups partitioned by result (hit or miss).
# TYPE modelops_inference_cache_requests_total counter
modelops_inference_cache_requests_total{model="iris_sentinel",result="hit",version="v1.0.0"} 32.0
modelops_inference_cache_requests_total{model="iris_sentinel",result="miss",version="v1.0.0"} 13.0
```

---

## 4. Grafana Dashboard Specification

The dashboard is pre-provisioned via `deploy/grafana/dashboards/modelops_sentinel.json` and automatically loaded at Grafana startup.

### Dashboard Panels & PromQL Queries

1. **Total Request Rate**:
   ```promql
   sum(rate(modelops_http_requests_total[1m]))
   ```
2. **HTTP Error Rate (%)**:
   ```promql
   (sum(rate(modelops_http_errors_total[1m])) / sum(rate(modelops_http_requests_total[1m]))) * 100 or on() vector(0)
   ```
3. **p50 Latency (Median)**:
   ```promql
   histogram_quantile(0.50, sum(rate(modelops_http_request_duration_seconds_bucket[1m])) by (le)) * 1000
   ```
4. **p95 Latency (SLA Threshold)**:
   ```promql
   histogram_quantile(0.95, sum(rate(modelops_http_request_duration_seconds_bucket[1m])) by (le)) * 1000
   ```
5. **p99 Latency (Worst-Case Tail)**:
   ```promql
   histogram_quantile(0.99, sum(rate(modelops_http_request_duration_seconds_bucket[1m])) by (le)) * 1000
   ```
6. **Model & Version Inference Usage**:
   ```promql
   sum(rate(modelops_model_inferences_total[1m])) by (model, version)
   ```
7. **Cache Hit Rate (%)**:
   ```promql
   (sum(rate(modelops_inference_cache_requests_total{result="hit"}[1m])) / sum(rate(modelops_inference_cache_requests_total[1m]))) * 100 or on() vector(0)
   ```

---

## 5. Verification & Testing

### 1. Automated Test Suite (Pytest)
Run the dedicated metrics test suite:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_metrics.py -v
```

### 2. Manual Verification Script (Zero-Docker Compatible)
Execute the complete end-to-end metrics simulation:
```powershell
.\.venv\Scripts\python.exe scripts/test_stage7_manual.py
```
This script exercises:
- HTTP GET requests and HTTP 404 error tracking.
- Cache miss on first inference, cache hit on repeat inference.
- Async inference job queueing and worker completion.
- Inspects `/metrics` and validates all 7 metric types.

---

## 6. Commands to Run & Open Platform

### Start the Application Locally (PowerShell)
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Open Endpoints in Browser (PowerShell)
```powershell
# Open Swagger Documentation
Start-Process "http://127.0.0.1:8000/docs"

# Open Raw Prometheus Metrics Endpoint
Start-Process "http://127.0.0.1:8000/metrics"

# Open Prometheus UI (When running Docker)
Start-Process "http://127.0.0.1:9090"

# Open Grafana Dashboard (When running Docker)
Start-Process "http://127.0.0.1:3000"
```

### Start Prometheus & Grafana via Docker Compose (Optional)
```powershell
docker compose up -d prometheus grafana
```
*(Default Grafana login: `admin` / `admin`)*
