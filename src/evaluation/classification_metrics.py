"""Classification / detection metrics for window-level anomaly alerts.

See AI_CONTEXT.md Section 15.1 (detection metrics) and Section 19
(`evaluation/classification_metrics.py` contract): Precision, Recall,
F1, ROC-AUC, false-alarm rate.

All functions here operate on already-computed window-level alert
decisions and/or continuous anomaly scores. They do not fit, calibrate,
or tune anything against these inputs — per AI_CONTEXT.md Section 17
(Rule 3), thresholds must never be chosen to optimize a metric computed
by this module on test data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_bool_array(values: Sequence[int] | Sequence[bool] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        raise ValueError("expected a non-empty array of labels/alerts")
    return arr.astype(bool)


def precision_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Precision = TP / (TP + FP), for binary window-level alerts.

    Args:
        y_true: Ground-truth binary labels (1 = anomalous window).
        y_pred: Predicted binary alerts (1 = alert raised).

    Returns:
        Precision in [0, 1]. Returns 0.0 (not NaN) when no positive
        predictions were made, so it composes safely with averaging code.
    """
    yt, yp = _as_bool_array(y_true), _as_bool_array(y_pred)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Recall (a.k.a. sensitivity / true-positive rate) = TP / (TP + FN)."""
    yt, yp = _as_bool_array(y_true), _as_bool_array(y_pred)
    tp = int(np.sum(yt & yp))
    fn = int(np.sum(yt & ~yp))
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Harmonic mean of precision and recall."""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def false_alarm_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """False-alarm rate = FP / (FP + TN), i.e. false-positive rate.

    This is the complement of specificity and is the quantity referenced
    by the conformal-calibration target in AI_CONTEXT.md Section G3
    (e.g. "<=5% false alarms").
    """
    yt, yp = _as_bool_array(y_true), _as_bool_array(y_pred)
    fp = int(np.sum(~yt & yp))
    tn = int(np.sum(~yt & ~yp))
    denom = fp + tn
    return fp / denom if denom > 0 else 0.0


def roc_auc_score(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """ROC-AUC computed via the Mann-Whitney U statistic (rank-based).

    Implemented without scikit-learn to keep this module dependency-light
    and independently testable; equivalent to
    `sklearn.metrics.roc_auc_score` for the binary case.

    Args:
        y_true: Ground-truth binary labels (1 = anomalous window).
        scores: Continuous anomaly scores (e.g. reconstruction error).
            Higher score = more anomalous.

    Returns:
        AUC in [0, 1]. Returns 0.5 (uninformative) if `y_true` contains
        only one class, since AUC is undefined in that case.
    """
    yt = _as_bool_array(y_true)
    s = np.asarray(scores, dtype=float)
    if yt.shape != s.shape:
        raise ValueError("y_true and scores must have the same shape")

    n_pos = int(np.sum(yt))
    n_neg = len(yt) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average ranks (ties split), 1-indexed, ascending by score.
    ranks = np.empty(len(s))
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]

    i = 0
    rank = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        rank += j - i + 1
        i = j + 1

    sum_ranks_pos = float(np.sum(ranks[yt]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


__all__ = [
    "precision_score",
    "recall_score",
    "f1_score",
    "false_alarm_rate",
    "roc_auc_score",
]
