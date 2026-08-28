"""Unit tests for `src/evaluation/*` metric modules.

See AI_CONTEXT.md Section 18 (`tests/` contract) and Section 19
(evaluation module contracts).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.classification_metrics import (
    f1_score,
    false_alarm_rate,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.evaluation.lead_time import detection_lead_time, first_valid_alert_cycle
from src.evaluation.nasa_score import nasa_score
from src.evaluation.rul_metrics import mean_absolute_error, root_mean_squared_error


# --- classification_metrics -------------------------------------------------


def test_precision_recall_f1_known_values():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]
    # TP=2, FP=1, FN=1, TN=1
    assert precision_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred) == pytest.approx(2 / 3)


def test_false_alarm_rate():
    y_true = [0, 0, 0, 1]
    y_pred = [1, 0, 0, 1]
    # FP=1, TN=2
    assert false_alarm_rate(y_true, y_pred) == pytest.approx(1 / 3)


def test_precision_zero_predictions_returns_zero_not_nan():
    assert precision_score([1, 0], [0, 0]) == 0.0


def test_roc_auc_perfect_separation():
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc_score(y_true, scores) == pytest.approx(1.0)


def test_roc_auc_single_class_returns_half():
    assert roc_auc_score([1, 1, 1], [0.1, 0.5, 0.9]) == 0.5


# --- rul_metrics -------------------------------------------------------------


def test_mae_rmse_known_values():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 33]
    # abs errors: 2, 2, 3 -> MAE = 7/3
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(7 / 3)
    # sq errors: 4, 4, 9 -> mean=17/3 -> sqrt
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(17 / 3))


def test_rmse_penalizes_large_errors_more_than_mae_ratio():
    y_true = [0, 0]
    y_pred = [1, 9]  # one small, one large error
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    assert rmse > mae


# --- nasa_score ----------------------------------------------------------


def test_nasa_score_zero_for_perfect_prediction():
    assert nasa_score([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)


def test_nasa_score_penalizes_late_more_than_early():
    # Same magnitude of error, opposite sign: late (over-prediction)
    # must score higher than early (under-prediction) with default denoms.
    early_score = nasa_score([50], [40])  # d = -10 (early)
    late_score = nasa_score([50], [60])  # d = +10 (late)
    assert late_score > early_score


def test_nasa_score_matches_manual_formula():
    y_true = [50]
    y_pred = [65]  # d = +15, late branch
    expected = np.exp(15 / 10) - 1
    assert nasa_score(y_true, y_pred) == pytest.approx(expected)


# --- lead_time -----------------------------------------------------------


def test_first_valid_alert_cycle_requires_persistence():
    cycles = [1, 2, 3, 4, 5, 6]
    alerts = [False, True, False, True, True, True]
    # persistence=1 -> first True at cycle 2
    assert first_valid_alert_cycle(cycles, alerts, persistence=1) == 2
    # persistence=3 -> needs cycles 4,5,6 -> starts at cycle 4
    assert first_valid_alert_cycle(cycles, alerts, persistence=3) == 4
    # persistence=4 -> never satisfied
    assert first_valid_alert_cycle(cycles, alerts, persistence=4) is None


def test_detection_lead_time_positive_when_alert_before_failure():
    cycles = [1, 2, 3, 4, 5]
    alerts = [False, False, True, True, True]
    result = detection_lead_time(cycles, alerts, failure_cycle=10, persistence=2)
    assert result.detected
    assert result.first_alert_cycle == 3
    assert result.lead_time == pytest.approx(7.0)


def test_detection_lead_time_missed_detection_returns_none():
    cycles = [1, 2, 3]
    alerts = [False, False, False]
    result = detection_lead_time(cycles, alerts, failure_cycle=10, persistence=1)
    assert not result.detected
    assert result.first_alert_cycle is None
    assert result.lead_time is None
