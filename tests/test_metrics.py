"""Unit and integration tests for Prometheus metrics collection and /metrics endpoint."""

import pytest
from httpx import AsyncClient

from app.core.metrics import (
    normalize_endpoint,
    track_async_job,
    track_cache_lookup,
    track_inference,
)


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_200_and_prometheus_format(
    async_client: AsyncClient,
) -> None:
    """Verify GET /metrics returns HTTP 200 with standard Prometheus exposition format."""
    response = await async_client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")

    content = response.text
    # Check for standard Prometheus comment markers
    assert "# HELP" in content
    assert "# TYPE" in content
    assert "modelops_http_requests_total" in content
    assert "modelops_http_request_duration_seconds" in content


@pytest.mark.asyncio
async def test_http_request_and_latency_tracked(
    async_client: AsyncClient,
) -> None:
    """Verify HTTP requests increment request counter and record latency histogram."""
    # Send a request to health endpoint
    health_resp = await async_client.get("/api/v1/health")
    assert health_resp.status_code == 200

    # Scrape metrics
    metrics_resp = await async_client.get("/metrics")
    assert metrics_resp.status_code == 200
    content = metrics_resp.text

    # Verify request counter with labels
    assert (
        'modelops_http_requests_total{endpoint="/api/v1/health",method="GET",status_code="200"}'
        in content
    )
    # Verify request latency histogram buckets and count
    assert (
        'modelops_http_request_duration_seconds_count{endpoint="/api/v1/health",method="GET"}'
        in content
    )


@pytest.mark.asyncio
async def test_http_errors_tracked(
    async_client: AsyncClient,
) -> None:
    """Verify HTTP error status codes (>= 400) increment error counters."""
    # Request a non-existent route to trigger 404
    resp = await async_client.get("/api/v1/nonexistent-endpoint-for-testing")
    assert resp.status_code == 404

    # Scrape metrics
    metrics_resp = await async_client.get("/metrics")
    content = metrics_resp.text

    assert (
        'modelops_http_errors_total{endpoint="/api/v1/nonexistent-endpoint-for-testing",method="GET",status_code="404"}'
        in content
    )


def test_endpoint_normalization_prevents_high_cardinality() -> None:
    """Verify normalize_endpoint replaces UUIDs with {id} placeholder."""
    raw_path_model = "/api/v1/models/a1b2c3d4-e5f6-7890-abcd-ef1234567890/predict"
    normalized_model = normalize_endpoint(raw_path_model)
    assert normalized_model == "/api/v1/models/{id}/predict"

    raw_path_job = "/api/v1/jobs/11223344-5566-7788-99aa-bbccddeeff00"
    normalized_job = normalize_endpoint(raw_path_job)
    assert normalized_job == "/api/v1/jobs/{id}"

    static_path = "/api/v1/health"
    assert normalize_endpoint(static_path) == "/api/v1/health"


def test_inference_and_cache_metrics() -> None:
    """Verify track_inference and track_cache_lookup record appropriate labels and values."""
    track_inference(
        model="churn_predictor",
        version="v1.0.0",
        duration_seconds=0.045,
        status="success",
    )
    track_cache_lookup(model="churn_predictor", version="v1.0.0", hit=True)
    track_cache_lookup(model="churn_predictor", version="v1.0.0", hit=False)

    from app.core.metrics import get_latest_metrics

    metrics_text = get_latest_metrics().decode("utf-8")

    # Verify model usage counter
    assert (
        'modelops_model_inferences_total{model="churn_predictor",status="success",version="v1.0.0"}'
        in metrics_text
    )
    # Verify inference duration histogram count
    assert (
        'modelops_inference_duration_seconds_count{model="churn_predictor",version="v1.0.0"}'
        in metrics_text
    )
    # Verify cache hit and miss counters
    assert (
        'modelops_inference_cache_requests_total{model="churn_predictor",result="hit",version="v1.0.0"}'
        in metrics_text
    )
    assert (
        'modelops_inference_cache_requests_total{model="churn_predictor",result="miss",version="v1.0.0"}'
        in metrics_text
    )


def test_async_job_metrics() -> None:
    """Verify track_async_job records background job lifecycle transitions."""
    track_async_job(job_type="async_predict_task", status="enqueued")
    track_async_job(job_type="async_predict_task", status="completed")
    track_async_job(job_type="process_telemetry_task", status="completed")

    from app.core.metrics import get_latest_metrics

    metrics_text = get_latest_metrics().decode("utf-8")

    assert (
        'modelops_async_jobs_total{job_type="async_predict_task",status="enqueued"}' in metrics_text
    )
    assert (
        'modelops_async_jobs_total{job_type="async_predict_task",status="completed"}'
        in metrics_text
    )
    assert (
        'modelops_async_jobs_total{job_type="process_telemetry_task",status="completed"}'
        in metrics_text
    )
