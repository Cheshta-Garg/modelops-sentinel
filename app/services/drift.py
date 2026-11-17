"""Statistical Drift Monitoring Service for ModelOps Sentinel.

This module implements:
1. Population Stability Index (PSI) algorithm with robust quantile binning.
2. Two-Sample Kolmogorov-Smirnov (KS) Test for continuous distributions.
3. Baseline reference distribution extraction and persistence.
4. Drift evaluation engine comparing production inferences against reference baselines.
5. Automatic recording of drift_events with feature, method, score, threshold, and severity.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.models.enums import DriftMetric
from app.models.inference import InferenceRequest
from app.models.metrics import ReferenceStats
from app.models.model import Model, ModelVersion
from app.models.monitoring import DriftEvent

logger = get_logger(__name__)

# Standard PSI thresholds
PSI_THRESHOLD_MODERATE = 0.10
PSI_THRESHOLD_SIGNIFICANT = 0.25

# Standard KS-Test threshold (significance level alpha)
KS_ALPHA_DEFAULT = 0.05
KS_STATISTIC_MODERATE = 0.10
KS_STATISTIC_SIGNIFICANT = 0.25


# ------------------------------------------------------------------------------
# Core Statistical Algorithms
# ------------------------------------------------------------------------------


def calculate_psi(
    reference: np.ndarray | list[float],
    current: np.ndarray | list[float],
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> tuple[float, dict[str, Any]]:
    """Compute the Population Stability Index (PSI) between reference and current samples.

    Formula:
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        where Expected% is the reference proportion and Actual% is the production proportion.

    Returns:
        tuple of (psi_score, details_dict)
    """
    ref_arr = np.asarray(reference, dtype=float)
    curr_arr = np.asarray(current, dtype=float)

    # Filter out NaNs / Infs
    ref_arr = ref_arr[np.isfinite(ref_arr)]
    curr_arr = curr_arr[np.isfinite(curr_arr)]

    if len(ref_arr) < 5 or len(curr_arr) < 5:
        return 0.0, {
            "error": "Insufficient samples for statistical PSI calculation (minimum 5 required)."
        }

    # Adapt bin count dynamically if sample size is small to prevent false positive sparsity spikes
    effective_bins = min(num_bins, max(4, len(curr_arr) // 8)) if len(curr_arr) < 80 else num_bins

    # Define bin edges using reference quantiles
    quantiles = np.linspace(0, 100, effective_bins + 1)
    bin_edges = np.percentile(ref_arr, quantiles)

    # De-duplicate bin edges if reference has repeated values
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        # Fallback to equal-width bins if all values identical
        min_v, max_v = float(ref_arr.min()), float(ref_arr.max())
        bin_edges = np.linspace(min_v - 1.0, max_v + 1.0, effective_bins + 1)

    # Expand outer boundaries to -inf and +inf to capture all production values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequency counts in each bin
    ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_arr, bins=bin_edges)
    b_count = len(ref_counts)

    # Apply Laplace pseudo-count smoothing: prevents zero-frequency division and false positives
    ref_pct = (ref_counts + 1) / (len(ref_arr) + b_count)
    curr_pct = (curr_counts + 1) / (len(curr_arr) + b_count)

    # Calculate PSI per bin
    bin_psi = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    total_psi = float(np.sum(bin_psi))

    details = {
        "num_bins": b_count,
        "ref_sample_count": len(ref_arr),
        "curr_sample_count": len(curr_arr),
        "bin_psi_contributions": [round(float(b), 5) for b in bin_psi],
        "ref_proportions": [round(float(p), 4) for p in ref_pct],
        "curr_proportions": [round(float(p), 4) for p in curr_pct],
    }

    return round(total_psi, 4), details


def calculate_ks_test(
    reference: np.ndarray | list[float],
    current: np.ndarray | list[float],
    alpha: float = KS_ALPHA_DEFAULT,
) -> tuple[float, float, bool]:
    """Perform a Two-Sample Kolmogorov-Smirnov Test between reference and current samples.

    H0 (Null Hypothesis): Reference and Current samples are drawn from the same continuous distribution.
    H1: The two distributions are different.

    Returns:
        tuple of (ks_statistic_D, p_value, drift_detected)
    """
    ref_arr = np.asarray(reference, dtype=float)
    curr_arr = np.asarray(current, dtype=float)

    ref_arr = ref_arr[np.isfinite(ref_arr)]
    curr_arr = curr_arr[np.isfinite(curr_arr)]

    if len(ref_arr) < 5 or len(curr_arr) < 5:
        return 0.0, 1.0, False

    # Two-sample KS test
    ks_res = stats.ks_2samp(ref_arr, curr_arr)
    statistic = round(float(ks_res.statistic), 4)
    p_value = round(float(ks_res.pvalue), 6)

    # Drift is rejected under H0 if p-value < alpha and effect size (D) > threshold
    drift_detected = bool(p_value < alpha and statistic >= KS_STATISTIC_MODERATE)

    return statistic, p_value, drift_detected


def classify_psi_severity(psi_score: float) -> str:
    """Classify drift severity according to industry standard PSI cutoffs."""
    if psi_score < PSI_THRESHOLD_MODERATE:
        return "NONE"
    if psi_score < PSI_THRESHOLD_SIGNIFICANT:
        return "MEDIUM"
    return "HIGH"


def classify_ks_severity(statistic: float, p_value: float) -> str:
    """Classify drift severity according to KS test statistic and significance."""
    if p_value >= KS_ALPHA_DEFAULT or statistic < KS_STATISTIC_MODERATE:
        return "NONE"
    if statistic < KS_STATISTIC_SIGNIFICANT:
        return "MEDIUM"
    return "HIGH"


# ------------------------------------------------------------------------------
# Baseline Reference Stats Utilities
# ------------------------------------------------------------------------------


def extract_reference_statistics(
    data: dict[str, list[float]] | np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Compute baseline statistical profile from model-version training/reference data."""
    feature_stats: dict[str, Any] = {}

    if isinstance(data, dict):
        for fname in feature_names:
            vals = np.asarray(data.get(fname, []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            feature_stats[fname] = _profile_feature_array(vals)
    elif isinstance(data, np.ndarray):
        for i, fname in enumerate(feature_names):
            if i < data.shape[1]:
                vals = data[:, i].astype(float)
                vals = vals[np.isfinite(vals)]
                feature_stats[fname] = _profile_feature_array(vals)

    return feature_stats


def _profile_feature_array(vals: np.ndarray) -> dict[str, Any]:
    """Extract descriptive statistics, decile quantiles, and representative sample values."""
    q_edges = np.percentile(vals, np.linspace(0, 100, 11))
    return {
        "count": int(len(vals)),
        "mean": round(float(np.mean(vals)), 4),
        "std": round(float(np.std(vals)), 4),
        "min": round(float(np.min(vals)), 4),
        "max": round(float(np.max(vals)), 4),
        "p25": round(float(np.percentile(vals, 25)), 4),
        "p50": round(float(np.median(vals)), 4),
        "p75": round(float(np.percentile(vals, 75)), 4),
        "bin_edges": [round(float(b), 4) for b in q_edges],
        "sample_values": [round(float(v), 4) for v in vals[:500]],  # Retain sample for KS testing
    }


# ------------------------------------------------------------------------------
# Drift Evaluation Service
# ------------------------------------------------------------------------------


class DriftService:
    """Service for managing reference baseline statistics, evaluating drift, and querying drift events."""

    @staticmethod
    async def save_reference_stats(
        session: AsyncSession,
        model_version_id: uuid.UUID,
        feature_data: dict[str, list[float]] | np.ndarray,
        feature_names: list[str],
        prediction_stats: dict[str, Any] | None = None,
    ) -> ReferenceStats:
        """Create or update baseline reference statistics for a specific model version."""
        stmt = select(ReferenceStats).where(ReferenceStats.model_version_id == model_version_id)
        existing = (await session.execute(stmt)).scalar_one_or_none()

        profile = extract_reference_statistics(feature_data, feature_names)
        sample_count = (
            len(feature_data)
            if isinstance(feature_data, (list, np.ndarray))
            else max(len(v) for v in feature_data.values())
            if isinstance(feature_data, dict) and feature_data
            else 0
        )

        if existing:
            existing.feature_stats = profile
            existing.num_samples = sample_count
            existing.prediction_stats = prediction_stats
            await session.commit()
            return existing

        ref = ReferenceStats(
            model_version_id=model_version_id,
            num_samples=sample_count,
            feature_stats=profile,
            prediction_stats=prediction_stats,
        )
        session.add(ref)
        await session.commit()
        return ref

    @staticmethod
    async def evaluate_drift(
        session: AsyncSession,
        model_version_id: uuid.UUID,
        method: str = "PSI",
        window_size: int = 100,
        threshold: float | None = None,
    ) -> list[DriftEvent]:
        """Evaluate incoming production feature distributions against baseline reference statistics.

        Records a DriftEvent for each evaluated feature.
        """
        method_norm = method.upper()
        if method_norm not in ("PSI", "KS_TEST"):
            method_norm = "PSI"

        # 1. Fetch Reference Statistics for the Model Version
        stmt_ref = select(ReferenceStats).where(ReferenceStats.model_version_id == model_version_id)
        ref_record = (await session.execute(stmt_ref)).scalar_one_or_none()
        if not ref_record or not ref_record.feature_stats:
            logger.warning(
                f"No reference statistics configured for model_version '{model_version_id}'. Cannot evaluate drift."
            )
            return []

        # 2. Fetch Latest Production Inferences for the Model Version
        stmt_inf = (
            select(InferenceRequest)
            .where(InferenceRequest.model_version_id == model_version_id)
            .order_by(desc(InferenceRequest.created_at))
            .limit(window_size)
        )
        inferences = (await session.execute(stmt_inf)).scalars().all()

        if len(inferences) < 5:
            logger.info(
                f"Insufficient production inference records ({len(inferences)} < 5) to evaluate drift for model_version '{model_version_id}'"
            )
            return []

        # Group production features
        prod_features: dict[str, list[float]] = {}
        window_end = inferences[0].created_at
        window_start = inferences[-1].created_at

        for inf in inferences:
            if isinstance(inf.input_features, dict):
                for k, v in inf.input_features.items():
                    try:
                        fval = float(v)
                        prod_features.setdefault(k, []).append(fval)
                    except (ValueError, TypeError):
                        continue

        drift_events_created: list[DriftEvent] = []
        now = datetime.now(timezone.utc)

        # 3. Evaluate each feature present in both Reference and Production
        for feature_name, ref_info in ref_record.feature_stats.items():
            curr_values = prod_features.get(feature_name)
            if not curr_values or len(curr_values) < 5:
                continue

            ref_sample = ref_info.get("sample_values")
            if not ref_sample:
                # If sample values weren't saved, reconstruct normal distribution approximation from mean/std
                mean_v = ref_info.get("mean", 0.0)
                std_v = ref_info.get("std", 1.0) or 1.0
                ref_sample = np.random.normal(
                    mean_v, std_v, size=min(100, len(curr_values))
                ).tolist()

            if method_norm == "PSI":
                used_threshold = threshold if threshold is not None else PSI_THRESHOLD_SIGNIFICANT
                score, details = calculate_psi(ref_sample, curr_values)
                drift_detected = score >= used_threshold
                severity = classify_psi_severity(score)
            else:  # KS_TEST
                used_threshold = threshold if threshold is not None else KS_STATISTIC_SIGNIFICANT
                stat_d, p_val, drift_detected = calculate_ks_test(ref_sample, curr_values)
                score = stat_d
                severity = classify_ks_severity(stat_d, p_val)
                details = {"ks_statistic": stat_d, "p_value": p_val, "alpha": KS_ALPHA_DEFAULT}

            # Create granular per-feature DriftEvent
            event = DriftEvent(
                model_version_id=model_version_id,
                feature=feature_name,
                method=method_norm,
                score=score,
                threshold=used_threshold,
                severity=severity,
                detected_at=now,
                drift_detected=drift_detected,
                window_start=window_start,
                window_end=window_end,
                metric_used=DriftMetric.PSI if method_norm == "PSI" else DriftMetric.KS_TEST,
                overall_drift_score=score,
                feature_drift_details=details,
                created_at=now,
            )
            session.add(event)
            drift_events_created.append(event)

        await session.commit()
        logger.info(
            f"Drift evaluation completed for model_version '{model_version_id}': {len(drift_events_created)} features checked, "
            f"{sum(1 for e in drift_events_created if e.drift_detected)} drifted."
        )
        return drift_events_created

    @staticmethod
    async def get_drift_events_for_model(
        session: AsyncSession,
        model_id: uuid.UUID,
        version_tag: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Query and summarize drift events for a given model."""
        stmt_model = select(Model).where(Model.id == model_id).options(selectinload(Model.versions))
        model = (await session.execute(stmt_model)).scalar_one_or_none()
        if not model:
            raise NotFoundError(message=f"Model with id '{model_id}' was not found.")

        # Resolve target version
        target_version: ModelVersion | None = None
        if version_tag:
            target_version = next((v for v in model.versions if v.version_tag == version_tag), None)
            if not target_version:
                raise NotFoundError(
                    message=f"Version '{version_tag}' not found for model '{model.name}'."
                )
        else:
            if not model.versions:
                return {
                    "model_id": str(model.id),
                    "model_name": model.name,
                    "active_version": None,
                    "drift_detected": False,
                    "summary": {
                        "total_events": 0,
                        "drifted_features_count": 0,
                        "highest_severity": "NONE",
                    },
                    "events": [],
                }
            # Pick latest or active version
            target_version = sorted(model.versions, key=lambda v: v.created_at, reverse=True)[0]

        # Check reference stats presence
        stmt_ref = select(ReferenceStats).where(
            ReferenceStats.model_version_id == target_version.id
        )
        ref_record = (await session.execute(stmt_ref)).scalar_one_or_none()

        # Query recent drift events
        stmt_events = (
            select(DriftEvent)
            .where(DriftEvent.model_version_id == target_version.id)
            .order_by(desc(DriftEvent.detected_at))
            .limit(limit)
        )
        events = (await session.execute(stmt_events)).scalars().all()

        drifted_count = sum(1 for e in events if e.drift_detected)
        severities = [e.severity for e in events if e.severity]
        highest_severity = (
            "HIGH" if "HIGH" in severities else "MEDIUM" if "MEDIUM" in severities else "NONE"
        )

        serialized_events = [
            {
                "id": str(e.id),
                "feature": e.feature or "overall",
                "method": e.method
                or (e.metric_used.value if hasattr(e.metric_used, "value") else str(e.metric_used)),
                "score": round(float(e.score if e.score is not None else e.overall_drift_score), 4),
                "threshold": round(float(e.threshold), 4),
                "severity": e.severity or ("HIGH" if e.drift_detected else "NONE"),
                "drift_detected": bool(e.drift_detected),
                "detected_at": e.detected_at.isoformat()
                if e.detected_at
                else e.created_at.isoformat(),
            }
            for e in events
        ]

        return {
            "model_id": str(model.id),
            "model_name": model.name,
            "version_tag": target_version.version_tag,
            "reference_stats_configured": ref_record is not None,
            "drift_detected": drifted_count > 0,
            "summary": {
                "total_evaluations": len(events),
                "drifted_features_count": drifted_count,
                "highest_severity": highest_severity,
            },
            "events": serialized_events,
        }


# Singleton service instance
drift_service = DriftService()
