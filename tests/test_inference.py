import io
import json
import uuid
from typing import Any

import joblib
import pytest
from httpx import AsyncClient
from sklearn.ensemble import RandomForestClassifier

from app.models.enums import ModelStatus
from app.services.inference import inference_engine


def create_joblib_bytes(obj: object) -> bytes:
    """Serialize a Python object into in-memory joblib bytes."""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


class MockBrokenModel:
    """Mock model that raises an internal computation exception during prediction."""

    def predict(self, x: Any) -> Any:
        raise RuntimeError("Simulated matrix singularity / memory corruption fault")


class InMemoryRedisMock:
    """In-memory Redis mock for unit testing cache hit/miss behavior."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self.store[key] = value
        return True

    async def lpush(self, key: str, value: str) -> int:
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].insert(0, value)
        return len(self.lists[key])

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def clean_model_cache():
    """Ensure in-process model cache is clean before each test."""
    inference_engine.invalidate_model_cache()
    yield
    inference_engine.invalidate_model_cache()


async def setup_swat_model(async_client: AsyncClient) -> tuple[str, str]:
    """Helper to register and upload a valid SWaT anomaly detection model."""
    create_resp = await async_client.post(
        "/api/v1/models",
        json={"name": f"swat_anomaly_{uuid.uuid4().hex[:8]}", "task_type": "binary_classification"},
    )
    assert create_resp.status_code == 201
    model_id = create_resp.json()["id"]

    # Train a minimal RandomForest on 5 SWaT features
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    # Fit dummy normal (0) and attack (1) samples
    X_dummy = [
        [650.0, 2.45, 255.0, 1, 19.5],  # normal
        [950.0, 5.50, 350.0, 0, 45.0],  # attack
    ]
    y_dummy = [0, 1]
    clf.fit(X_dummy, y_dummy)

    artifact_bytes = create_joblib_bytes(clf)
    input_schema = {
        "features": ["LIT101", "FIT101", "AIT201", "P101", "DPIT301"],
        "types": {
            "LIT101": "float",
            "FIT101": "float",
            "AIT201": "float",
            "P101": "int",
            "DPIT301": "float",
        },
    }
    output_schema = {"prediction": "int", "confidence": "float"}

    files = {"file": ("model.joblib", artifact_bytes, "application/octet-stream")}
    data = {
        "version_tag": "v1.0.0",
        "input_schema": json.dumps(input_schema),
        "output_schema": json.dumps(output_schema),
        "artifact_format": "JOBLIB",
    }
    ver_resp = await async_client.post(f"/api/v1/models/{model_id}/versions", files=files, data=data)
    assert ver_resp.status_code == 201
    assert ver_resp.json()["status"] == ModelStatus.STAGING

    return model_id, "v1.0.0"


@pytest.mark.asyncio
async def test_valid_inference_with_dict_features(async_client: AsyncClient) -> None:
    """Verify POST /api/v1/models/{id}/predict returns correct schema, confidence, and latency."""
    model_id, version_tag = await setup_swat_model(async_client)

    payload = {
        "features": {
            "LIT101": 650.0,
            "FIT101": 2.45,
            "AIT201": 255.0,
            "P101": 1,
            "DPIT301": 19.5,
        },
        "version_tag": version_tag,
    }
    response = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "request_id" in data and len(data["request_id"]) > 0
    assert "swat_anomaly_" in data["model"]
    assert data["version"] == "v1.0.0"
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] > 0
    assert data["cached"] is False


@pytest.mark.asyncio
async def test_valid_inference_with_list_features(async_client: AsyncClient) -> None:
    """Verify inference accepts ordered list feature format."""
    model_id, _ = await setup_swat_model(async_client)

    payload = {
        "features": [650.0, 2.45, 255.0, 1, 19.5],
    }
    response = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert data["version"] == "v1.0.0"


@pytest.mark.asyncio
async def test_inference_invalid_input_missing_features(async_client: AsyncClient) -> None:
    """Verify 422 error when required input features are omitted."""
    model_id, _ = await setup_swat_model(async_client)

    payload = {
        "features": {"LIT101": 650.0},  # Missing FIT101, AIT201, P101, DPIT301
    }
    response = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "INVALID_INPUT_FEATURES"
    assert "Missing required feature(s)" in data["error"]["message"]


@pytest.mark.asyncio
async def test_inference_invalid_input_wrong_list_length(async_client: AsyncClient) -> None:
    """Verify 422 error when feature vector length does not match expected schema."""
    model_id, _ = await setup_swat_model(async_client)

    payload = {"features": [650.0, 2.45]}  # Only 2 elements, expected 5
    response = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "INVALID_INPUT_FEATURES"


@pytest.mark.asyncio
async def test_inference_missing_model_returns_404(async_client: AsyncClient) -> None:
    """Verify 404 error when querying an un-registered model UUID."""
    random_id = uuid.uuid4()
    payload = {"features": {"LIT101": 650.0, "FIT101": 2.45, "AIT201": 255.0, "P101": 1, "DPIT301": 19.5}}
    response = await async_client.post(f"/api/v1/models/{random_id}/predict", json=payload)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_inference_missing_version_returns_404(async_client: AsyncClient) -> None:
    """Verify 404 error when requesting an invalid version tag."""
    model_id, _ = await setup_swat_model(async_client)
    payload = {
        "features": {"LIT101": 650.0, "FIT101": 2.45, "AIT201": 255.0, "P101": 1, "DPIT301": 19.5},
        "version_tag": "v9.9.9",
    }
    response = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_inference_cache_miss_then_cache_hit(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify first request results in cache miss and second identical request hits Redis cache."""
    model_id, _ = await setup_swat_model(async_client)
    mock_redis = InMemoryRedisMock()
    monkeypatch.setattr("app.services.inference.get_redis_client", lambda: mock_redis)

    payload = {
        "features": {
            "LIT101": 650.0,
            "FIT101": 2.45,
            "AIT201": 255.0,
            "P101": 1,
            "DPIT301": 19.5,
        }
    }

    # Request 1: Cache Miss
    resp1 = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["cached"] is False
    assert len(mock_redis.store) == 1
    assert "telemetry:buffer" in mock_redis.lists

    # Request 2: Cache Hit
    resp2 = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["cached"] is True
    assert data2["prediction"] == data1["prediction"]
    assert data2["confidence"] == data1["confidence"]


@pytest.mark.asyncio
async def test_inference_cache_invalidation(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify clearing cache forces a re-computation of inference."""
    model_id, _ = await setup_swat_model(async_client)
    mock_redis = InMemoryRedisMock()
    monkeypatch.setattr("app.services.inference.get_redis_client", lambda: mock_redis)

    payload = {
        "features": {
            "LIT101": 650.0,
            "FIT101": 2.45,
            "AIT201": 255.0,
            "P101": 1,
            "DPIT301": 19.5,
        }
    }

    # First call: miss
    r1 = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert r1.json()["cached"] is False

    # Second call: hit
    r2 = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert r2.json()["cached"] is True

    # Invalidate Redis store & in-memory cache
    mock_redis.store.clear()
    inference_engine.invalidate_model_cache()

    # Third call: miss again
    r3 = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert r3.json()["cached"] is False


@pytest.mark.asyncio
async def test_inference_execution_error_handled(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify runtime model compute exceptions return 500 INFERENCE_EXECUTION_FAILED."""
    model_id, _ = await setup_swat_model(async_client)

    # Monkeypatch the loaded model to return a broken instance
    broken_model = MockBrokenModel()
    monkeypatch.setattr(inference_engine, "_get_loaded_model", lambda ver: broken_model)

    payload = {
        "features": {
            "LIT101": 650.0,
            "FIT101": 2.45,
            "AIT201": 255.0,
            "P101": 1,
            "DPIT301": 19.5,
        }
    }
    response = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "INFERENCE_EXECUTION_FAILED"
    assert "Simulated matrix singularity" in data["error"]["message"]


@pytest.mark.asyncio
async def test_inference_redis_fail_open_resilience(
    async_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify inference succeeds in-memory when Redis raises an unhandled connection error."""
    model_id, _ = await setup_swat_model(async_client)

    class BrokenRedis:
        async def get(self, key: str) -> None:
            raise ConnectionError("Simulated Redis timeout/crash")

        async def setex(self, key: str, ttl: int, val: str) -> None:
            raise ConnectionError("Simulated Redis write timeout")

        async def lpush(self, key: str, val: str) -> None:
            raise ConnectionError("Simulated Redis queue failure")

    monkeypatch.setattr("app.services.inference.get_redis_client", lambda: BrokenRedis())

    payload = {
        "features": {
            "LIT101": 650.0,
            "FIT101": 2.45,
            "AIT201": 255.0,
            "P101": 1,
            "DPIT301": 19.5,
        }
    }
    response = await async_client.post(f"/api/v1/models/{model_id}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert data["cached"] is False
