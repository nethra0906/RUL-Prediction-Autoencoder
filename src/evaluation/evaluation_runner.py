"""Orchestrates a full evaluation run and result-table assembly.

See AI_CONTEXT.md Section 15 (evaluation design) and Section 19
(`evaluation/evaluation_runner.py` module contract).

Ground-truth anomaly labeling for the TEST set (AI_CONTEXT.md does not
prescribe one convention; this project's choice, decided 30 Aug 2026):
a test-set cycle is labeled anomalous if it falls beyond
`healthy_frac` of that engine's TRUE total life, where true total life
is reconstructed from the ground-truth test RUL file:

    total_life = last_observed_cycle_in_test + true_RUL

This mirrors the closest prior-art paper's "last 10% of life is
degraded" heuristic, but ties the cutoff to the same `healthy_frac`
config value already used to select the healthy training region
(AI_CONTEXT.md Section 5.3), so the healthy/degraded boundary is
defined consistently on both sides of the pipeline rather than as two
independently-chosen numbers.

This is evaluation-only labeling — it uses the ground-truth test RUL
file exclusively to construct labels for scoring, never as model input
or for threshold selection (AI_CONTEXT.md Section 17 Rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.classification_metrics import (
    f1_score,
    false_alarm_rate,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.evaluation.lead_time import LeadTimeResult, detection_lead_time


def label_anomalous_by_life_fraction(
    test_df: pd.DataFrame,
    test_rul: pd.Series,
    healthy_frac: float,
    engine_col: str = "unit_id",
    cycle_col: str = "cycle",
    output_col: str = "is_anomalous",
) -> pd.DataFrame:
    """Label test-set rows anomalous/healthy using true total life.

    For each engine, true total life is `last_observed_cycle + true_RUL`
    (from `test_rul`, indexed by engine ID — see `data/loaders.py`'s
    `load_test_rul`). A row is anomalous if its cycle exceeds
    `healthy_frac * total_life`.

    Args:
        test_df: Truncated test-trajectory DataFrame (e.g. from
            `load_test`), containing at least `engine_col`/`cycle_col`.
        test_rul: Ground-truth RUL at truncation, indexed by engine ID
            (e.g. from `load_test_rul`). Every engine in `test_df` must
            have a corresponding entry.
        healthy_frac: Same fraction used for training healthy-region
            selection (`data/healthy_region.py`), so both boundaries
            are defined consistently. Must be in (0, 1).
        engine_col: Column identifying the engine/unit.
        cycle_col: Column identifying the cycle index.
        output_col: Name of the boolean label column to add.

    Returns:
        A copy of `test_df` with an added boolean `output_col`.

    Raises:
        ValueError: If `healthy_frac` is not in (0, 1), required
            columns are missing, or an engine in `test_df` has no
            entry in `test_rul`.
    """
    if not (0.0 < healthy_frac < 1.0):
        raise ValueError(f"healthy_frac must be in (0, 1), got {healthy_frac!r}")

    missing = {engine_col, cycle_col} - set(test_df.columns)
    if missing:
        raise ValueError(f"test_df is missing required column(s): {sorted(missing)}")

    engine_ids = test_df[engine_col].unique()
    missing_rul = set(engine_ids) - set(test_rul.index)
    if missing_rul:
        raise ValueError(f"test_rul is missing entries for engine(s): {sorted(missing_rul)}")

    result = test_df.copy()
    last_observed_cycle = result.groupby(engine_col)[cycle_col].transform("max")
    true_rul_per_row = result[engine_col].map(test_rul)
    total_life = last_observed_cycle + true_rul_per_row

    result[output_col] = result[cycle_col] > (healthy_frac * total_life)
    return result


def compute_engine_total_life(
    test_df: pd.DataFrame,
    test_rul: pd.Series,
    engine_col: str = "unit_id",
    cycle_col: str = "cycle",
) -> pd.Series:
    """Per-engine true total life: last_observed_cycle + true_RUL.

    Used as the `failure_cycle` input to `evaluation.lead_time` for
    each engine — see `evaluate_detection` below.

    Returns:
        Series indexed by engine ID, values are total life (int-like).
    """
    last_cycle = test_df.groupby(engine_col)[cycle_col].max()
    missing_rul = set(last_cycle.index) - set(test_rul.index)
    if missing_rul:
        raise ValueError(f"test_rul is missing entries for engine(s): {sorted(missing_rul)}")
    return last_cycle + test_rul.reindex(last_cycle.index)


@dataclass(frozen=True)
class DetectionEvaluationResult:
    """Full evaluation summary for one experiment run.

    Attributes:
        precision, recall, f1, roc_auc, false_alarm_rate: Window-level
            classification metrics (AI_CONTEXT.md Section 15.1).
        detection_rate: Fraction of engines with at least one valid
            (persistence-satisfying) alert before their observed
            truncation point.
        mean_lead_time: Mean lead time among DETECTED engines only
            (missed detections excluded, per AI_CONTEXT.md Section 13 —
            averaging in a sentinel for missed detections would distort
            the metric). None if no engine was detected.
        per_engine_results: LeadTimeResult per engine, keyed by engine ID.
        n_engines: Total engines evaluated.
    """

    precision: float
    recall: float
    f1: float
    roc_auc: float
    false_alarm_rate: float
    detection_rate: float
    mean_lead_time: Optional[float]
    per_engine_results: dict
    n_engines: int


def evaluate_detection(
    y_true: np.ndarray,
    scores: np.ndarray,
    alerts: np.ndarray,
    engine_ids: np.ndarray,
    end_cycles: np.ndarray,
    engine_total_life: pd.Series,
    persistence: int = 1,
) -> DetectionEvaluationResult:
    """Compute the full detection metric suite for one experiment run.

    Args:
        y_true: Boolean/int window-level ground-truth labels (from
            `label_anomalous_by_life_fraction`, taken at each window's
            end cycle — see `data/windows.py`'s `label_cols`).
        scores: Continuous window-level anomaly scores (e.g. from
            `anomaly.reconstruction.window_scores_numpy`).
        alerts: Boolean window-level alert decisions (e.g. from
            `anomaly.fixed_threshold.apply_fixed_threshold`).
        engine_ids: Engine ID per window (from `WindowedSequences.engine_ids`).
        end_cycles: End cycle per window (from `WindowedSequences.end_cycles`).
        engine_total_life: Per-engine true total life (from
            `compute_engine_total_life`), used as each engine's
            `failure_cycle` for lead-time computation.
        persistence: Consecutive-alert rule passed to
            `evaluation.lead_time.detection_lead_time`. AI_CONTEXT.md
            Section 12: never invent this silently in general, but a
            default of 1 (first crossing) is used here explicitly as
            the not-yet-specified starting point — override once the
            team decides on a real persistence value.

    Returns:
        A `DetectionEvaluationResult` with window-level classification
        metrics plus per-engine, event-level lead-time results.

    Raises:
        ValueError: If array lengths are mismatched, or any engine_id
            in `engine_ids` has no entry in `engine_total_life`.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    alerts = np.asarray(alerts).astype(bool)
    engine_ids = np.asarray(engine_ids)
    end_cycles = np.asarray(end_cycles)

    lengths = {len(y_true), len(scores), len(alerts), len(engine_ids), len(end_cycles)}
    if len(lengths) != 1:
        raise ValueError(
            f"y_true/scores/alerts/engine_ids/end_cycles must all have the same "
            f"length, got lengths {lengths}"
        )

    missing = set(np.unique(engine_ids)) - set(engine_total_life.index)
    if missing:
        raise ValueError(f"engine_total_life is missing entries for engine(s): {sorted(missing)}")

    precision = precision_score(y_true, alerts)
    recall = recall_score(y_true, alerts)
    f1 = f1_score(y_true, alerts)
    auc = roc_auc_score(y_true, scores)
    far = false_alarm_rate(y_true, alerts)

    per_engine_results: dict[object, LeadTimeResult] = {}
    for engine_id in np.unique(engine_ids):
        mask = engine_ids == engine_id
        order = np.argsort(end_cycles[mask])
        cycles = end_cycles[mask][order].tolist()
        engine_alerts = alerts[mask][order].tolist()
        failure_cycle = int(engine_total_life.loc[engine_id])

        per_engine_results[engine_id] = detection_lead_time(
            cycles, engine_alerts, failure_cycle, persistence
        )

    detected_results = [r for r in per_engine_results.values() if r.detected]
    detection_rate = len(detected_results) / len(per_engine_results) if per_engine_results else 0.0
    mean_lead_time = (
        float(np.mean([r.lead_time for r in detected_results])) if detected_results else None
    )

    return DetectionEvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=auc,
        false_alarm_rate=far,
        detection_rate=detection_rate,
        mean_lead_time=mean_lead_time,
        per_engine_results=per_engine_results,
        n_engines=len(per_engine_results),
    )


__all__ = [
    "label_anomalous_by_life_fraction",
    "compute_engine_total_life",
    "DetectionEvaluationResult",
    "evaluate_detection",
]
