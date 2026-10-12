import io
import json
import uuid

import joblib
import pytest
from httpx import AsyncClient
from sklearn.linear_model import LogisticRegression

from app.models.enums import ModelStatus


class DummyClassifier:
    """Minimal mock model implementing callable predict method."""

    def predict(self, x: list) -> list:
        return [1 for _ in x]


def create_joblib_bytes(obj: object) -> bytes:
    """Serialize a Python object into in-memory joblib bytes."""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_register_model_success(async_client: AsyncClient) -> None:
    """Verify POST /api/v1/models registers a new logical model."""
    payload = {
        "name": "churn_predictor_v1",
        "description": "Predicts customer subscription churn",
        "task_type": "binary_classification",
    }
    response = await async_client.post("/api/v1/models", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "churn_predictor_v1"
    assert data["task_type"] == "binary_classification"
    assert "id" in data
    assert data["versions_count"] == 0


@pytest.mark.asyncio
async def test_register_model_duplicate_name_conflict(async_client: AsyncClient) -> None:
    """Verify duplicate model name returns 409 Conflict."""
    payload = {
        "name": "fraud_detector",
        "description": "Initial registration",
        "task_type": "classification",
    }
    resp1 = await async_client.post("/api/v1/models", json=payload)
    assert resp1.status_code == 201

    resp2 = await async_client.post("/api/v1/models", json=payload)
    assert resp2.status_code == 409
    data = resp2.json()
    assert data["error"]["code"] == "MODEL_ALREADY_EXISTS"


@pytest.mark.asyncio
async def test_list_models(async_client: AsyncClient) -> None:
    """Verify GET /api/v1/models lists registered models."""
    await async_client.post("/api/v1/models", json={"name": "model_alpha", "task_type": "reg"})
    await async_client.post("/api/v1/models", json={"name": "model_beta", "task_type": "clf"})

    response = await async_client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    names = [m["name"] for m in data]
    assert "model_alpha" in names
    assert "model_beta" in names


@pytest.mark.asyncio
async def test_get_model_by_id_and_not_found(async_client: AsyncClient) -> None:
    """Verify GET /api/v1/models/{id} returns model or 404."""
    create_resp = await async_client.post(
        "/api/v1/models",
        json={"name": "target_model", "task_type": "clf"},
    )
    model_id = create_resp.json()["id"]

    get_resp = await async_client.get(f"/api/v1/models/{model_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "target_model"
    assert get_resp.json()["versions"] == []

    # Non-existent ID returns 404
    non_existent = str(uuid.uuid4())
    err_resp = await async_client.get(f"/api/v1/models/{non_existent}")
    assert err_resp.status_code == 404
    assert err_resp.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_upload_valid_joblib_artifact_pipeline(async_client: AsyncClient) -> None:
    """Verify full artifact upload pipeline: validates, hashes, stores, and transitions to STAGING."""
    create_resp = await async_client.post(
        "/api/v1/models",
        json={"name": "valid_pipeline_model", "task_type": "clf"},
    )
    model_id = create_resp.json()["id"]

    # Create real valid scikit-learn model
    lr = LogisticRegression()
    # Fit dummy data so model has classes_
    lr.fit([[0, 0], [1, 1]], [0, 1])
    artifact_bytes = create_joblib_bytes(lr)

    files = {
        "file": ("model.joblib", artifact_bytes, "application/octet-stream"),
    }
    data = {
        "version_tag": "v1.0.0",
        "input_schema": json.dumps({"features": ["feature_1", "feature_2"]}),
        "output_schema": json.dumps({"prediction": "int", "probabilities": "list"}),
        "artifact_format": "JOBLIB",
    }

    response = await async_client.post(
        f"/api/v1/models/{model_id}/versions",
        files=files,
        data=data,
    )
    assert response.status_code == 201
    ver_data = response.json()
    assert ver_data["version_tag"] == "v1.0.0"
    assert ver_data["status"] == ModelStatus.STAGING
    assert len(ver_data["artifact_hash_sha256"]) == 64
    assert "model.joblib" in ver_data["artifact_path"]

    # Verify model detail now shows this version in summary
    model_detail = await async_client.get(f"/api/v1/models/{model_id}")
    assert model_detail.json()["versions_count"] == 1
    assert model_detail.json()["versions"][0]["version_tag"] == "v1.0.0"


@pytest.mark.asyncio
async def test_upload_version_duplicate_tag_conflict(async_client: AsyncClient) -> None:
    """Verify uploading duplicate version tag returns 409 Conflict."""
    create_resp = await async_client.post(
        "/api/v1/models",
        json={"name": "dup_tag_model", "task_type": "clf"},
    )
    model_id = create_resp.json()["id"]

    artifact_bytes = create_joblib_bytes(DummyClassifier())
    files = {"file": ("model.joblib", artifact_bytes, "application/octet-stream")}
    form_data = {
        "version_tag": "v1.0.0",
        "input_schema": json.dumps({"features": ["x"]}),
        "output_schema": json.dumps({"target": "y"}),
    }

    resp1 = await async_client.post(f"/api/v1/models/{model_id}/versions", files=files, data=form_data)
    assert resp1.status_code == 201

    # Repeat with same version tag
    files2 = {"file": ("model.joblib", artifact_bytes, "application/octet-stream")}
    resp2 = await async_client.post(f"/api/v1/models/{model_id}/versions", files=files2, data=form_data)
    assert resp2.status_code == 409
    assert resp2.json()["error"]["code"] == "VERSION_TAG_EXISTS"


@pytest.mark.asyncio
async def test_upload_invalid_extension_rejected(async_client: AsyncClient) -> None:
    """Verify uploading invalid file extension (e.g. .exe or .txt) returns 400 Bad Request."""
    create_resp = await async_client.post(
        "/api/v1/models",
        json={"name": "bad_ext_model", "task_type": "clf"},
    )
    model_id = create_resp.json()["id"]

    files = {"file": ("malicious_payload.exe", b"MZexecutabledata", "application/octet-stream")}
    form_data = {
        "version_tag": "v1.0.0",
        "input_schema": json.dumps({"features": ["x"]}),
        "output_schema": json.dumps({"target": "y"}),
    }

    response = await async_client.post(f"/api/v1/models/{model_id}/versions", files=files, data=form_data)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_ARTIFACT_EXTENSION"


@pytest.mark.asyncio
async def test_upload_corrupted_model_fails_validation(async_client: AsyncClient) -> None:
    """Verify corrupted binary fails load test, transitions to FAILED, and returns 422."""
    create_resp = await async_client.post(
        "/api/v1/models",
        json={"name": "corrupt_test_model", "task_type": "clf"},
    )
    model_id = create_resp.json()["id"]

    corrupted_bytes = b"NOT_A_VALID_JOBLIB_HEADER_GARBAGE_BYTES_12345"
    files = {"file": ("model.joblib", corrupted_bytes, "application/octet-stream")}
    form_data = {
        "version_tag": "v1.0.0",
        "input_schema": json.dumps({"features": ["x"]}),
        "output_schema": json.dumps({"target": "y"}),
    }

    response = await async_client.post(f"/api/v1/models/{model_id}/versions", files=files, data=form_data)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MODEL_VALIDATION_FAILED"


@pytest.mark.asyncio
async def test_upload_artifact_without_predict_method_fails(async_client: AsyncClient) -> None:
    """Verify serialized object lacking a callable predict() method fails validation (422)."""
    create_resp = await async_client.post(
        "/api/v1/models",
        json={"name": "no_predict_model", "task_type": "clf"},
    )
    model_id = create_resp.json()["id"]

    # Arbitrary object without predict method
    non_model_object = {"key": "this is just a python dictionary"}
    artifact_bytes = create_joblib_bytes(non_model_object)

    files = {"file": ("model.joblib", artifact_bytes, "application/octet-stream")}
    form_data = {
        "version_tag": "v1.0.0",
        "input_schema": json.dumps({"features": ["x"]}),
        "output_schema": json.dumps({"target": "y"}),
    }

    response = await async_client.post(f"/api/v1/models/{model_id}/versions", files=files, data=form_data)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MODEL_VALIDATION_FAILED"


@pytest.mark.asyncio
async def test_upload_malformed_schema_json_fails(async_client: AsyncClient) -> None:
    """Verify malformed JSON in input_schema or output_schema returns 400 Bad Request."""
    create_resp = await async_client.post(
        "/api/v1/models",
        json={"name": "malformed_schema_model", "task_type": "clf"},
    )
    model_id = create_resp.json()["id"]

    artifact_bytes = create_joblib_bytes(DummyClassifier())
    files = {"file": ("model.joblib", artifact_bytes, "application/octet-stream")}
    form_data = {
        "version_tag": "v1.0.0",
        "input_schema": "NOT_VALID_JSON{:::broken",
        "output_schema": json.dumps({"target": "y"}),
    }

    response = await async_client.post(f"/api/v1/models/{model_id}/versions", files=files, data=form_data)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MALFORMED_SCHEMA_JSON"
