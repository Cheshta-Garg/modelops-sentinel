"""Delayed Ground-Truth Feedback & Performance Evaluation Service for ModelOps Sentinel.

This module implements Stage 9:
1. Ingesting delayed real-world ground-truth labels matched to prior inference requests via request_id.
2. Computing true production classification performance metrics (accuracy, precision, recall, F1,
   and confusion matrix) exclusively from matched (prediction, ground_truth) pairs.
3. Persisting delayed ground-truth performance into `perf_metrics` while keeping it strictly
   separate from `offline_metrics` (pre-deployment validation) and online operational metrics
   (Prometheus / real-time inference telemetry).
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.errors import AppException, NotFoundError
from app.core.logging import get_logger
from app.models.enums import ModelStatus
from app.models.inference import GroundTruth, InferenceRequest
from app.models.metrics import PerfMetrics
from app.models.model import Model, ModelVersion
from app.schemas.ground_truth import FeedbackCreateRequest

logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# Label Normalization & Classification Metric Calculation
# ------------------------------------------------------------------------------


def _ensure_utc(dt: datetime) -> datetime:
    """Attach UTC timezone to naive datetimes returned by SQLite."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def extract_label_value(payload: Any) -> Any:
    """Extract canonical scalar label from JSONB prediction or ground_truth payloads."""
    if isinstance(payload, dict):
        if "value" in payload:
            return payload["value"]
        if "label" in payload:
            return payload["label"]
        if "prediction" in payload:
            return payload["prediction"]
    return payload


def _is_int_like(value: Any) -> bool:
    """Return True if value is a bool, int, or integral float (e.g. 1.0)."""
    if isinstance(value, (bool, int)):
        return True
    if isinstance(value, float) and value.is_integer():
        return True
    return False


def _normalize_labels(y_true: list[Any], y_pred: list[Any]) -> tuple[list[Any], list[Any]]:
    """Align ground-truth and predicted label types so sklearn metrics never fail on type mismatch."""
    combined = y_true + y_pred
    if all(_is_int_like(v) for v in combined):
        return [int(v) for v in y_true], [int(v) for v in y_pred]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in combined):
        return [float(v) for v in y_true], [float(v) for v in y_pred]
    return [str(v) for v in y_true], [str(v) for v in y_pred]


def compute_classification_metrics(
    y_true: list[Any],
    y_pred: list[Any],
    positive_label: Any | None = None,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, and structured confusion matrix.

    Supports both binary classification (producing explicit tp/tn/fp/fn counts) and
    multiclass classification (using weighted averaging across classes).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have equal length (got {len(y_true)} vs {len(y_pred)})."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute classification metrics on an empty sample set.")

    y_true_norm, y_pred_norm = _normalize_labels(y_true, y_pred)
    unique_labels = sorted(set(y_true_norm) | set(y_pred_norm))

    acc = float(accuracy_score(y_true_norm, y_pred_norm))

    # Binary classification with canonical {0, 1} labels
    if set(unique_labels).issubset({0, 1}):
        ordered_labels: list[Any] = [0, 1]
        pos_lbl = 1 if positive_label is None else int(positive_label)
        if pos_lbl not in (0, 1):
            pos_lbl = 1

        prec = float(precision_score(y_true_norm, y_pred_norm, pos_label=pos_lbl, zero_division=0))
        rec = float(recall_score(y_true_norm, y_pred_norm, pos_label=pos_lbl, zero_division=0))
        f1 = float(f1_score(y_true_norm, y_pred_norm, pos_label=pos_lbl, zero_division=0))

        cm = confusion_matrix(y_true_norm, y_pred_norm, labels=ordered_labels)
        if pos_lbl == 1:
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        else:
            tp, fn, fp, tn = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        cm_payload: dict[str, Any] = {
            "labels": ordered_labels,
            "matrix": [[int(c) for c in row] for row in cm.tolist()],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "averaging": "binary",
        }
    elif len(unique_labels) == 2:
        # Generic binary classification (e.g. ["benign", "fraud"] or [-1, 1])
        norm_pos = None
        if positive_label is not None:
            norm_pos_list, _ = _normalize_labels([positive_label], [unique_labels[0]])
            if norm_pos_list[0] in unique_labels:
                norm_pos = norm_pos_list[0]
        pos_lbl = norm_pos if norm_pos is not None else unique_labels[1]
        neg_lbl = unique_labels[0] if unique_labels[1] == pos_lbl else unique_labels[1]
        ordered_labels = [neg_lbl, pos_lbl]

        prec = float(precision_score(y_true_norm, y_pred_norm, pos_label=pos_lbl, zero_division=0))
        rec = float(recall_score(y_true_norm, y_pred_norm, pos_label=pos_lbl, zero_division=0))
        f1 = float(f1_score(y_true_norm, y_pred_norm, pos_label=pos_lbl, zero_division=0))

        cm = confusion_matrix(y_true_norm, y_pred_norm, labels=ordered_labels)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        cm_payload = {
            "labels": ordered_labels,
            "matrix": [[int(c) for c in row] for row in cm.tolist()],
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "averaging": "binary",
        }
    else:
        # Multiclass (or single non-binary class) classification
        ordered_labels = unique_labels
        prec = float(
            precision_score(y_true_norm, y_pred_norm, average="weighted", zero_division=0)
        )
        rec = float(recall_score(y_true_norm, y_pred_norm, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true_norm, y_pred_norm, average="weighted", zero_division=0))

        cm = confusion_matrix(y_true_norm, y_pred_norm, labels=ordered_labels)
        cm_payload = {
            "labels": ordered_labels,
            "matrix": [[int(c) for c in row] for row in cm.tolist()],
            "averaging": "weighted",
        }

    return {
        "sample_count": len(y_true_norm),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm_payload,
    }


# ------------------------------------------------------------------------------
# Ground-Truth Feedback & Delayed Performance Service
# ------------------------------------------------------------------------------


class GroundTruthService:
    """Service managing delayed ground-truth ingestion and performance metric computation."""

    @staticmethod
    async def record_feedback(
        session: AsyncSession,
        model_id: uuid.UUID,
        payload: FeedbackCreateRequest,
    ) -> dict[str, Any]:
        """Match a delayed ground-truth label to a prior inference request and store in ground_truth."""
        # 1. Verify logical model exists
        stmt_model = select(Model).where(Model.id == model_id)
        model = (await session.execute(stmt_model)).scalar_one_or_none()
        if not model:
            raise NotFoundError(message=f"Model with id '{model_id}' was not found.")

        # 2. Locate the InferenceRequest belonging to this model by request_id
        stmt_inf = (
            select(InferenceRequest, ModelVersion)
            .join(ModelVersion, InferenceRequest.model_version_id == ModelVersion.id)
            .where(
                ModelVersion.model_id == model_id,
                InferenceRequest.request_id == payload.request_id,
            )
            .order_by(desc(InferenceRequest.created_at))
            .limit(1)
        )
        row = (await session.execute(stmt_inf)).first()
        if not row:
            raise NotFoundError(
                message=(
                    f"Inference request '{payload.request_id}' was not found for model '{model.name}'."
                )
            )

        inf_request, version = row

        # 3. Guard against duplicate ground-truth submission for the same inference request
        stmt_existing = select(GroundTruth).where(
            GroundTruth.inference_request_id == inf_request.id
        )
        existing_gt = (await session.execute(stmt_existing)).scalar_one_or_none()
        if existing_gt is not None:
            raise AppException(
                message=(
                    f"Ground-truth feedback has already been recorded for request_id '{payload.request_id}'."
                ),
                code="DUPLICATE_FEEDBACK",
                status_code=409,
            )

        # 4. Persist immutable GroundTruth record
        now = datetime.now(timezone.utc)
        gt_record = GroundTruth(
            inference_request_id=inf_request.id,
            actual_label={"value": payload.label},
            matched_at=now,
            created_at=now,
        )
        session.add(gt_record)
        await session.commit()
        await session.refresh(gt_record)

        # 5. Compare normalized prediction vs actual label for immediate response clarity
        predicted_value = extract_label_value(inf_request.prediction)
        norm_true, norm_pred = _normalize_labels([payload.label], [predicted_value])
        is_correct = norm_true[0] == norm_pred[0]

        delay_seconds = max(
            0.0,
            round((_ensure_utc(now) - _ensure_utc(inf_request.created_at)).total_seconds(), 3),
        )

        logger.info(
            f"Recorded ground-truth feedback: model={model.name} version={version.version_tag} "
            f"request_id={payload.request_id} predicted={predicted_value} actual={payload.label} "
            f"delay={delay_seconds}s"
        )

        return {
            "id": gt_record.id,
            "model_id": model.id,
            "model_version_id": version.id,
            "version_tag": version.version_tag,
            "inference_request_id": inf_request.id,
            "request_id": inf_request.request_id,
            "predicted_label": predicted_value,
            "actual_label": payload.label,
            "is_correct": is_correct,
            "feedback_delay_seconds": delay_seconds,
            "matched_at": gt_record.matched_at,
        }

    @staticmethod
    async def evaluate_delayed_performance(
        session: AsyncSession,
        model_version_id: uuid.UUID,
        window_size: int = 500,
        since: datetime | None = None,
        until: datetime | None = None,
        positive_label: Any | None = None,
    ) -> PerfMetrics | None:
        """Join inference_requests with ground_truth and persist delayed performance into perf_metrics.

        Strictly writes ONLY delayed ground-truth classification metrics to `perf_metrics`.
        Never mutates `offline_metrics` or `online_metrics` and never mixes operational latency.
        """
        stmt_ver = select(ModelVersion).where(ModelVersion.id == model_version_id)
        version = (await session.execute(stmt_ver)).scalar_one_or_none()
        if not version:
            raise NotFoundError(
                message=f"Model version with id '{model_version_id}' was not found."
            )

        # Fetch matched (InferenceRequest, GroundTruth) pairs for this model version
        stmt = (
            select(InferenceRequest, GroundTruth)
            .join(GroundTruth, GroundTruth.inference_request_id == InferenceRequest.id)
            .where(InferenceRequest.model_version_id == model_version_id)
        )
        if since is not None:
            stmt = stmt.where(InferenceRequest.created_at >= since)
        if until is not None:
            stmt = stmt.where(InferenceRequest.created_at <= until)

        stmt = stmt.order_by(desc(InferenceRequest.created_at)).limit(window_size)
        matched_rows = (await session.execute(stmt)).all()

        if not matched_rows:
            logger.info(
                f"No matched ground-truth records available for model_version '{model_version_id}'. Skipping evaluation."
            )
            return None

        y_true: list[Any] = []
        y_pred: list[Any] = []
        timestamps: list[datetime] = []

        for inf_req, gt_rec in matched_rows:
            pred_val = extract_label_value(inf_req.prediction)
            actual_val = extract_label_value(gt_rec.actual_label)
            if pred_val is None or actual_val is None:
                continue
            y_pred.append(pred_val)
            y_true.append(actual_val)
            timestamps.append(_ensure_utc(inf_req.created_at))

        if not y_true:
            logger.warning(
                f"Matched rows for model_version '{model_version_id}' had no extractable label pairs."
            )
            return None

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            positive_label=positive_label,
        )

        now = datetime.now(timezone.utc)
        window_start = min(timestamps)
        window_end = max(timestamps)

        perf_record = PerfMetrics(
            model_version_id=model_version_id,
            window_start=window_start,
            window_end=window_end,
            sample_count=metrics["sample_count"],
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            confusion_matrix=metrics["confusion_matrix"],
            evaluated_at=now,
            created_at=now,
        )
        session.add(perf_record)
        await session.commit()
        await session.refresh(perf_record)

        logger.info(
            f"Delayed ground-truth performance evaluated for model_version '{model_version_id}': "
            f"n={perf_record.sample_count} acc={perf_record.accuracy} prec={perf_record.precision} "
            f"rec={perf_record.recall} f1={perf_record.f1_score}"
        )
        return perf_record

    @staticmethod
    async def get_model_performance(
        session: AsyncSession,
        model_id: uuid.UUID,
        version_tag: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Retrieve delayed ground-truth performance history and label coverage for a model."""
        stmt_model = select(Model).where(Model.id == model_id).options(selectinload(Model.versions))
        model = (await session.execute(stmt_model)).scalar_one_or_none()
        if not model:
            raise NotFoundError(message=f"Model with id '{model_id}' was not found.")

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
            if not model.versions:
                return {
                    "metric_category": "delayed_ground_truth_performance",
                    "model_id": model.id,
                    "model_name": model.name,
                    "version_tag": None,
                    "total_inference_requests": 0,
                    "matched_ground_truth_count": 0,
                    "label_coverage_ratio": 0.0,
                    "latest_metrics": None,
                    "history": [],
                }
            eligible = [
                v for v in model.versions if v.status in (ModelStatus.ACTIVE, ModelStatus.STAGING)
            ]
            pool = eligible if eligible else model.versions
            target_version = sorted(pool, key=lambda v: v.created_at, reverse=True)[0]

        # Count total inference requests and matched ground-truth labels for label coverage visibility
        total_inf_stmt = select(func.count(InferenceRequest.id)).where(
            InferenceRequest.model_version_id == target_version.id
        )
        total_inferences = int((await session.execute(total_inf_stmt)).scalar() or 0)

        matched_gt_stmt = (
            select(func.count(GroundTruth.id))
            .join(InferenceRequest, GroundTruth.inference_request_id == InferenceRequest.id)
            .where(InferenceRequest.model_version_id == target_version.id)
        )
        matched_gt_count = int((await session.execute(matched_gt_stmt)).scalar() or 0)

        coverage = round(matched_gt_count / total_inferences, 4) if total_inferences > 0 else 0.0

        # Query ONLY delayed ground-truth PerfMetrics rows (never operational/legacy rows)
        stmt_perf = (
            select(PerfMetrics)
            .where(
                PerfMetrics.model_version_id == target_version.id,
                PerfMetrics.sample_count > 0,
                PerfMetrics.accuracy.is_not(None),
            )
            .order_by(desc(PerfMetrics.evaluated_at))
            .limit(limit)
        )
        records = (await session.execute(stmt_perf)).scalars().all()

        history = [
            {
                "id": rec.id,
                "model_version_id": rec.model_version_id,
                "window_start": rec.window_start,
                "window_end": rec.window_end,
                "sample_count": rec.sample_count,
                "accuracy": float(rec.accuracy),
                "precision": float(rec.precision or 0.0),
                "recall": float(rec.recall or 0.0),
                "f1_score": float(rec.f1_score or 0.0),
                "confusion_matrix": rec.confusion_matrix or {"labels": [], "matrix": []},
                "evaluated_at": rec.evaluated_at,
            }
            for rec in records
        ]

        return {
            "metric_category": "delayed_ground_truth_performance",
            "model_id": model.id,
            "model_name": model.name,
            "version_tag": target_version.version_tag,
            "total_inference_requests": total_inferences,
            "matched_ground_truth_count": matched_gt_count,
            "label_coverage_ratio": coverage,
            "latest_metrics": history[0] if history else None,
            "history": history,
        }


# Singleton service instance
ground_truth_service = GroundTruthService()
