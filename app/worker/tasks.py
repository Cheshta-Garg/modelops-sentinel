import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from arq.worker import Retry
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.errors import AppException, NotFoundError
from app.core.logging import get_logger, request_id_ctx
from app.core.metrics import track_inference
from app.core.redis import get_redis_client
from app.db.session import AsyncSessionLocal
from app.models.enums import ModelStatus
from app.models.inference import InferenceRequest
from app.models.model import Model, ModelVersion
from app.services.inference import inference_engine

logger = get_logger(__name__)


async def async_predict_task(
    ctx: dict,
    model_id: str,
    payload: dict[str, Any],
    request_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Execute asynchronous machine learning inference in background worker."""
    req_id = request_id or str(ctx.get("job_id", uuid.uuid4().hex))
    token = request_id_ctx.set(req_id)
    t_start = time.perf_counter()

    job_try = ctx.get("job_try", 1)
    max_tries = ctx.get("max_tries", 3)
    logger.info(
        f"[Worker] Starting async inference job for model '{model_id}' (try {job_try}/{max_tries}, req_id: {req_id})"
    )

    try:
        model_uuid = uuid.UUID(str(model_id))
    except ValueError as exc:
        raise AppException(
            message=f"Invalid model UUID format: '{model_id}'",
            code="INVALID_MODEL_ID",
            status_code=400,
        ) from exc

    try:
        session_factory = ctx.get("session_factory") or AsyncSessionLocal
        async with session_factory() as session:
            # 1. Fetch Model and Target Version
            stmt = select(Model).where(Model.id == model_uuid).options(selectinload(Model.versions))
            model = (await session.execute(stmt)).scalar_one_or_none()
            if not model:
                raise NotFoundError(message=f"Model with id '{model_id}' was not found.")

            version_tag = payload.get("version_tag")
            target_version: ModelVersion | None = None

            if version_tag:
                target_version = next(
                    (v for v in model.versions if v.version_tag == version_tag), None
                )
                if not target_version:
                    raise NotFoundError(
                        message=f"Version '{version_tag}' not found for model '{model.name}'."
                    )
            else:
                eligible = [
                    v
                    for v in model.versions
                    if v.status in (ModelStatus.ACTIVE, ModelStatus.STAGING)
                ]
                if not eligible:
                    raise AppException(
                        message=f"Model '{model.name}' has no active or staging versions deployed for inference.",
                        code="NO_SERVICABLE_VERSION",
                        status_code=400,
                    )
                target_version = sorted(eligible, key=lambda v: v.created_at, reverse=True)[0]

            if target_version.status not in (ModelStatus.ACTIVE, ModelStatus.STAGING):
                raise AppException(
                    message=f"Version '{target_version.version_tag}' has status '{target_version.status.value}'. Only STAGING or ACTIVE can serve traffic.",
                    code="UNAPPROVED_VERSION_STATUS",
                    status_code=400,
                )

            # 2. Format & Validate Features
            features_input = payload.get("features")
            if features_input is None:
                raise AppException(
                    message="Payload missing 'features' field.",
                    code="MISSING_FEATURES",
                    status_code=422,
                )

            X_input, features_dict = inference_engine._format_features(
                features_input, target_version.input_schema
            )

            # 3. Model Retrieval & Inference Execution
            model_obj = inference_engine._get_loaded_model(target_version)
            raw_pred = model_obj.predict(X_input)[0]
            prediction: Any = (
                int(raw_pred)
                if isinstance(raw_pred, (np.integer, int))
                else float(raw_pred)
                if isinstance(raw_pred, (np.floating, float))
                else str(raw_pred)
            )

            confidence = 1.0
            if hasattr(model_obj, "predict_proba") and callable(model_obj.predict_proba):
                probas = model_obj.predict_proba(X_input)[0]
                confidence = float(np.max(probas))

            t_end = time.perf_counter()
            latency_ms = round((t_end - t_start) * 1000, 2)
            track_inference(
                model=model.name,
                version=target_version.version_tag,
                duration_seconds=(t_end - t_start),
                status="async_success",
            )

            # 4. Telemetry Creation & DB Persistence
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

            # Optional Redis buffer telemetry
            try:
                redis_client = get_redis_client()
                telemetry_event = {
                    "request_id": req_id,
                    "model_id": str(model.id),
                    "model_name": model.name,
                    "version_tag": target_version.version_tag,
                    "input_features": features_dict,
                    "prediction": prediction,
                    "confidence": round(confidence, 4),
                    "latency_ms": latency_ms,
                    "source": "async_worker",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await redis_client.lpush("telemetry:buffer", json.dumps(telemetry_event))
            except Exception as exc:
                logger.debug(f"[Worker] Redis telemetry buffer unavailable: {exc}")

            logger.info(
                f"[Worker] Async inference completed successfully: model={model.name}, version={target_version.version_tag}, pred={prediction}, latency={latency_ms}ms"
            )

            return {
                "request_id": req_id,
                "model": model.name,
                "version": target_version.version_tag,
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "latency_ms": latency_ms,
                "cached": False,
                "status": "COMPLETED",
            }
    except (NotFoundError, AppException, asyncio.CancelledError):
        # Client, validation, or cancellation errors should not trigger retries
        raise
    except Exception as exc:
        logger.error(f"[Worker] Unexpected error executing async inference: {exc}", exc_info=True)
        if job_try < max_tries:
            logger.warning(f"[Worker] Retrying async inference task ({job_try}/{max_tries})...")
            raise Retry(defer=2) from exc
        raise
    finally:
        request_id_ctx.reset(token)


async def process_telemetry_task(
    ctx: dict,
    telemetry_batch: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    """Process, aggregate, and summarize telemetry events buffered by inference endpoints."""
    items = telemetry_batch if isinstance(telemetry_batch, list) else [telemetry_batch]
    count = len(items)
    latencies = [item.get("latency_ms", 0.0) for item in items if "latency_ms" in item]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    logger.info(
        f"[Worker] Processing telemetry batch of {count} events (avg latency: {avg_latency}ms)"
    )

    # Group predictions by class/value
    distribution: dict[str, int] = {}
    for item in items:
        pred_key = str(item.get("prediction", "unknown"))
        distribution[pred_key] = distribution.get(pred_key, 0) + 1

    return {
        "processed_count": count,
        "avg_latency_ms": avg_latency,
        "prediction_distribution": distribution,
        "status": "COMPLETED",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


async def evaluate_drift_task(
    ctx: dict,
    model_version_id: str,
    method: str = "PSI",
    window_size: int = 100,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Background task to periodically evaluate feature drift against reference statistics."""
    logger.info(
        f"[Worker] Starting drift evaluation task for model_version '{model_version_id}' using method '{method}'"
    )
    session_factory = ctx.get("session_factory") or AsyncSessionLocal
    try:
        version_uuid = uuid.UUID(str(model_version_id))
    except ValueError as exc:
        raise AppException(
            message=f"Invalid model_version UUID format: '{model_version_id}'",
            code="INVALID_UUID",
            status_code=400,
        ) from exc

    async with session_factory() as session:
        from app.services.drift import drift_service

        events = await drift_service.evaluate_drift(
            session=session,
            model_version_id=version_uuid,
            method=method,
            window_size=window_size,
            threshold=threshold,
        )

        drifted = [e for e in events if e.drift_detected]
        result = {
            "model_version_id": str(model_version_id),
            "method": method.upper(),
            "total_features_evaluated": len(events),
            "drifted_features_count": len(drifted),
            "drift_detected": len(drifted) > 0,
            "drifted_features": [
                {
                    "feature": e.feature,
                    "score": round(float(e.score), 4) if e.score is not None else 0.0,
                    "threshold": round(float(e.threshold), 4),
                    "severity": e.severity,
                }
                for e in drifted
            ],
            "status": "COMPLETED",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            f"[Worker] Drift evaluation finished: {len(events)} evaluated, {len(drifted)} drifted ({[e.feature for e in drifted]})"
        )
        return result


TASK_REGISTRY = {
    "async_predict_task": async_predict_task,
    "process_telemetry_task": process_telemetry_task,
    "evaluate_drift_task": evaluate_drift_task,
}
