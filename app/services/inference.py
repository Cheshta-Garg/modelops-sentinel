import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.errors import AppException, NotFoundError
from app.core.logging import get_logger, request_id_ctx
from app.core.redis import get_redis_client
from app.models.enums import ModelStatus
from app.models.inference import InferenceRequest
from app.models.model import Model, ModelVersion
from app.schemas.inference import InferenceRequestPayload, InferenceResponse
from app.storage.artifact_store import LocalArtifactStore

logger = get_logger(__name__)

# Cache TTL for repeated prediction results (5 minutes)
RESULT_CACHE_TTL_SECONDS = 300


class InferenceEngine:
    """Synchronous Model Serving Engine with in-process caching, Redis result caching, and telemetry."""

    def __init__(self, storage: LocalArtifactStore | None = None) -> None:
        self.storage = storage or LocalArtifactStore()
        # In-process model cache: maps version_id (UUID string) -> loaded model instance
        self._model_cache: dict[str, Any] = {}
        # In-memory result cache fallback (for local standalone testing without Redis)
        self._memory_result_cache: dict[str, str] = {}

    def _get_loaded_model(self, version: ModelVersion) -> Any:
        """Retrieve model object from in-process memory cache or load from disk artifact."""
        cache_key = str(version.id)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        abs_path = self.storage.get_absolute_path(version.artifact_path)
        if not abs_path.exists():
            raise NotFoundError(message=f"Model artifact file not found on disk at {version.artifact_path}")

        try:
            model_obj = joblib.load(abs_path)
            self._model_cache[cache_key] = model_obj
            logger.info(f"Loaded and cached model artifact in-process: {version.version_tag} ({version.id})")
            return model_obj
        except Exception as exc:
            logger.error(f"Failed to deserialize model artifact for version {version.id}: {exc}", exc_info=True)
            raise AppException(
                message=f"Failed to load model artifact: {exc}",
                code="MODEL_LOAD_ERROR",
                status_code=500,
            ) from exc

    def invalidate_model_cache(self, version_id: uuid.UUID | None = None) -> None:
        """Evict loaded model from in-memory cache (or clear all if version_id is None)."""
        if version_id:
            self._model_cache.pop(str(version_id), None)
        else:
            self._model_cache.clear()
            self._memory_result_cache.clear()


    @staticmethod
    def _compute_feature_cache_key(
        model_id: uuid.UUID,
        version_tag: str,
        features: dict[str, Any] | list[Any],
    ) -> str:
        """Generate a deterministic SHA-256 hash key for input features."""
        canonical_repr = json.dumps(features, sort_keys=True, default=str)
        feature_hash = hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()[:16]
        return f"cache:inf:{model_id}:{version_tag}:{feature_hash}"

    @staticmethod
    def _format_features(
        features: dict[str, Any] | list[Any],
        input_schema: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Validate feature schema and construct a 2D numpy array for scikit-learn."""
        expected_features: list[str] = input_schema.get("features", [])

        if isinstance(features, dict):
            if expected_features:
                missing = [f for f in expected_features if f not in features]
                if missing:
                    raise AppException(
                        message=f"Missing required feature(s): {missing}",
                        code="INVALID_INPUT_FEATURES",
                        status_code=422,
                        details={"missing_features": missing, "expected": expected_features},
                    )
                ordered_values = [float(features[f]) for f in expected_features]
                features_dict = {f: features[f] for f in expected_features}
            else:
                ordered_values = [float(v) for v in features.values()]
                features_dict = features
            return np.array([ordered_values], dtype=float), features_dict

        if isinstance(features, list):
            if expected_features and len(features) != len(expected_features):
                raise AppException(
                    message=f"Feature length mismatch: expected {len(expected_features)}, got {len(features)}",
                    code="INVALID_INPUT_FEATURES",
                    status_code=422,
                )
            features_dict = {f: features[i] for i, f in enumerate(expected_features)} if expected_features else {"raw": features}
            return np.array([features], dtype=float), features_dict

        raise AppException(
            message="Input features must be a dictionary or list.",
            code="INVALID_INPUT_FEATURES",
            status_code=422,
        )

    async def predict(
        self,
        session: AsyncSession,
        model_id: uuid.UUID,
        payload: InferenceRequestPayload,
    ) -> InferenceResponse:
        """Serve synchronous prediction with caching, latency timing, and telemetry logging."""
        t_start = time.perf_counter()
        req_id = request_id_ctx.get()
        if req_id == "-":
            req_id = uuid.uuid4().hex

        # 1. Fetch Model and Target Version
        stmt = (
            select(Model)
            .where(Model.id == model_id)
            .options(selectinload(Model.versions))
        )
        model = (await session.execute(stmt)).scalar_one_or_none()
        if not model:
            raise NotFoundError(message=f"Model with id '{model_id}' was not found.")

        # Resolve version: requested tag or highest validated version (ACTIVE or STAGING)
        target_version: ModelVersion | None = None
        if payload.version_tag:
            target_version = next((v for v in model.versions if v.version_tag == payload.version_tag), None)
            if not target_version:
                raise NotFoundError(message=f"Version '{payload.version_tag}' not found for model '{model.name}'.")
        else:
            eligible = [v for v in model.versions if v.status in (ModelStatus.ACTIVE, ModelStatus.STAGING)]
            if not eligible:
                raise AppException(
                    message=f"Model '{model.name}' has no active or staging versions deployed for inference.",
                    code="NO_SERVICABLE_VERSION",
                    status_code=400,
                )
            # Pick latest eligible version
            target_version = sorted(eligible, key=lambda v: v.created_at, reverse=True)[0]

        # Enforce lifecycle state requirement
        if target_version.status not in (ModelStatus.ACTIVE, ModelStatus.STAGING):
            raise AppException(
                message=f"Version '{target_version.version_tag}' has status '{target_version.status.value}'. Only STAGING or ACTIVE can serve traffic.",
                code="UNAPPROVED_VERSION_STATUS",
                status_code=400,
            )

        # 2. Check Redis Result Cache (Fail-Open)
        cache_key = self._compute_feature_cache_key(model.id, target_version.version_tag, payload.features)
        redis_client = get_redis_client()
        cached_result: str | None = None
        try:
            cached_result = await redis_client.get(cache_key)
        except Exception as exc:
            logger.debug(f"Redis cache lookup failed (fail-open): {exc}")
            cached_result = self._memory_result_cache.get(cache_key)


        if cached_result:
            try:
                cached_data = json.loads(cached_result)
                t_end = time.perf_counter()
                latency_ms = round((t_end - t_start) * 1000, 2)
                return InferenceResponse(
                    request_id=req_id,
                    model=model.name,
                    version=target_version.version_tag,
                    prediction=cached_data["prediction"],
                    confidence=cached_data["confidence"],
                    latency_ms=latency_ms,
                    cached=True,
                )
            except Exception as exc:
                logger.warning(f"Corrupt cached prediction payload, falling back to inference: {exc}")

        # 3. Format & Validate Features
        X_input, features_dict = self._format_features(payload.features, target_version.input_schema)

        # 4. In-Process Model Retrieval & Inference Execution
        model_obj = self._get_loaded_model(target_version)
        try:
            raw_pred = model_obj.predict(X_input)[0]
            # Convert numpy types to native Python
            prediction: Any = int(raw_pred) if isinstance(raw_pred, (np.integer, int)) else float(raw_pred) if isinstance(raw_pred, (np.floating, float)) else str(raw_pred)

            confidence = 1.0
            if hasattr(model_obj, "predict_proba") and callable(model_obj.predict_proba):
                probas = model_obj.predict_proba(X_input)[0]
                confidence = float(np.max(probas))
        except Exception as exc:
            logger.error(f"Inference computation error for {model.name}:{target_version.version_tag}: {exc}", exc_info=True)
            raise AppException(
                message=f"Model inference failed during calculation: {exc}",
                code="INFERENCE_EXECUTION_FAILED",
                status_code=500,
            ) from exc

        t_end = time.perf_counter()
        latency_ms = round((t_end - t_start) * 1000, 2)

        # 5. Store in Result Cache (Memory Fallback + Redis Fail-Open)
        cache_payload = json.dumps({"prediction": prediction, "confidence": round(confidence, 4)})
        self._memory_result_cache[cache_key] = cache_payload
        try:
            await redis_client.setex(cache_key, RESULT_CACHE_TTL_SECONDS, cache_payload)
        except Exception as exc:
            logger.debug(f"Failed to cache inference result in Redis (fail-open): {exc}")


        # 6. Telemetry Creation: PostgreSQL Audit Record & Redis Queue Buffer
        inf_record = InferenceRequest(
            request_id=req_id,
            model_version_id=target_version.id,
            input_features=features_dict,
            prediction={"value": prediction, "confidence": round(confidence, 4)},
            latency_ms=latency_ms,
            cached=False,
        )
        session.add(inf_record)
        await session.commit()

        # Push to Redis buffer for downstream async worker telemetry processing
        try:
            telemetry_event = {
                "request_id": req_id,
                "model_id": str(model.id),
                "model_name": model.name,
                "version_tag": target_version.version_tag,
                "input_features": features_dict,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "latency_ms": latency_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await redis_client.lpush("telemetry:buffer", json.dumps(telemetry_event))
        except Exception as exc:
            logger.debug(f"Failed to buffer telemetry event to Redis (fail-open): {exc}")

        return InferenceResponse(
            request_id=req_id,
            model=model.name,
            version=target_version.version_tag,
            prediction=prediction,
            confidence=round(confidence, 4),
            latency_ms=latency_ms,
            cached=False,
        )


# Singleton engine instance
inference_engine = InferenceEngine()

