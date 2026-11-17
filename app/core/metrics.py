"""Prometheus metrics instrumentation for ModelOps Sentinel.

This module defines the Prometheus metric collectors (Counters, Histograms)
and helper methods for recording production telemetry across HTTP requests,
inference execution, cache performance, and asynchronous background jobs.

Crucial Architecture Constraint:
Prometheus metrics are kept purely in-memory as ephemeral time-series data
scraped via HTTP GET /metrics. They are NOT stored in application relational
databases (PostgreSQL / SQLite) as application history.
"""

import re

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

# ------------------------------------------------------------------------------
# Metric Definitions
# ------------------------------------------------------------------------------

# 1. HTTP Request Count
http_requests_total = Counter(
    name="modelops_http_requests_total",
    documentation="Total count of HTTP requests processed by the platform.",
    labelnames=["method", "endpoint", "status_code"],
)

# 2. HTTP Error Count
http_errors_total = Counter(
    name="modelops_http_errors_total",
    documentation="Total count of HTTP error responses (HTTP status >= 400).",
    labelnames=["method", "endpoint", "status_code"],
)

# 3. HTTP Request Latency (seconds)
http_request_duration_seconds = Histogram(
    name="modelops_http_request_duration_seconds",
    documentation="HTTP request latency in seconds from receipt to response.",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# 4. Model Inference Latency (seconds)
inference_duration_seconds = Histogram(
    name="modelops_inference_duration_seconds",
    documentation="Pure model inference computation latency in seconds (excluding queue/network).",
    labelnames=["model", "version"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# 5. Model / Version Usage Count
model_inferences_total = Counter(
    name="modelops_model_inferences_total",
    documentation="Total number of inference predictions performed per model and version.",
    labelnames=["model", "version", "status"],
)

# 6. Async Background Job Count
async_jobs_total = Counter(
    name="modelops_async_jobs_total",
    documentation="Total count of background async jobs processed or enqueued.",
    labelnames=["job_type", "status"],
)

# 7. Prediction Result Cache Hit / Miss Count
inference_cache_requests_total = Counter(
    name="modelops_inference_cache_requests_total",
    documentation="Inference result cache lookups partitioned by result (hit or miss).",
    labelnames=["model", "version", "result"],
)


# ------------------------------------------------------------------------------
# Helpers for Metric Recording & Label Cardinality Control
# ------------------------------------------------------------------------------

# Regex to match UUIDs and 24-36 char hex hashes to prevent label cardinality explosion
_ID_REGEX = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-fA-F]{24,36}"
)


def normalize_endpoint(path: str) -> str:
    """Normalize route path to prevent label cardinality explosion.

    Replaces dynamic UUIDs, entity IDs, and hex hashes with '{id}' while
    preserving the full URL path hierarchy (e.g. '/api/v1/models/{id}/predict').
    """
    return _ID_REGEX.sub("{id}", path)


def track_http_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Record HTTP request count, latency histogram, and error counters."""
    status_str = str(status_code)
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_str,
    ).inc()

    http_request_duration_seconds.labels(
        method=method,
        endpoint=endpoint,
    ).observe(duration_seconds)

    if status_code >= 400:
        http_errors_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_str,
        ).inc()


def track_inference(
    model: str,
    version: str,
    duration_seconds: float,
    status: str = "success",
) -> None:
    """Record pure model inference latency and usage counters."""
    inference_duration_seconds.labels(
        model=model,
        version=version,
    ).observe(duration_seconds)

    model_inferences_total.labels(
        model=model,
        version=version,
        status=status,
    ).inc()


def track_cache_lookup(model: str, version: str, hit: bool) -> None:
    """Record cache hit or miss for model inference result cache."""
    result_str = "hit" if hit else "miss"
    inference_cache_requests_total.labels(
        model=model,
        version=version,
        result=result_str,
    ).inc()


def track_async_job(job_type: str, status: str) -> None:
    """Record asynchronous background job submission and completion states."""
    async_jobs_total.labels(
        job_type=job_type,
        status=status,
    ).inc()


def get_latest_metrics(registry: CollectorRegistry = REGISTRY) -> bytes:
    """Generate Prometheus exposition text format payload for /metrics scraping."""
    return generate_latest(registry)
