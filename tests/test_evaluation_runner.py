"""Tests for `src/evaluation/evaluation_runner.py`.

See AI_CONTEXT.md Section 17 Rule 5 (test RUL is evaluation-only) and
Section 15 (evaluation design).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluation_runner import (
    compute_engine_total_life,
    evaluate_detection,
    label_anomalous_by_life_fraction,
)


def _make_test_df(engine_id: int, n_observed_cycles: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [engine_id] * n_observed_cycles,
            "cycle": list(range(1, n_observed_cycles + 1)),
        }
    )


# --- label_anomalous_by_life_fraction ---------------------------------------


def test_labels_beyond_healthy_frac_of_true_total_life():
    # Engine observed for 80 cycles, true RUL=20 -> total_life=100.
    # healthy_frac=0.85 -> cutoff at cycle 85. Since only 80 cycles are
    # observed, nothing in this truncated data should be anomalous yet.
    df = _make_test_df(engine_id=1, n_observed_cycles=80)
    rul = pd.Series({1: 20}, name="RUL")

    result = label_anomalous_by_life_fraction(df, rul, healthy_frac=0.85)

    assert not result["is_anomalous"].any()


def test_labels_anomalous_when_truncated_past_cutoff():
    # Engine observed for 95 cycles, true RUL=5 -> total_life=100.
    # cutoff at cycle 85 -> cycles 86-95 are anomalous (10 rows).
    df = _make_test_df(engine_id=1, n_observed_cycles=95)
    rul = pd.Series({1: 5}, name="RUL")

    result = label_anomalous_by_life_fraction(df, rul, healthy_frac=0.85)

    assert result["is_anomalous"].sum() == 10
    assert result[result["cycle"] <= 85]["is_anomalous"].sum() == 0
    assert result[result["cycle"] > 85]["is_anomalous"].all()


def test_multiple_engines_use_own_total_life():
    df = pd.concat(
        [_make_test_df(1, n_observed_cycles=95), _make_test_df(2, n_observed_cycles=40)],
        ignore_index=True,
    )
    rul = pd.Series({1: 5, 2: 5}, name="RUL")  # engine1 total=100, engine2 total=45

    result = label_anomalous_by_life_fraction(df, rul, healthy_frac=0.85)

    engine1_anomalous = result[result["unit_id"] == 1]["is_anomalous"].sum()
    engine2_anomalous = result[result["unit_id"] == 2]["is_anomalous"].sum()

    assert engine1_anomalous == 10  # cycles 86-95
    # engine2: cutoff = 0.85*45 = 38.25 -> cycles 39,40 anomalous
    assert engine2_anomalous == 2


def test_missing_rul_entry_raises():
    df = _make_test_df(engine_id=1, n_observed_cycles=50)
    rul = pd.Series({2: 10}, name="RUL")  # wrong engine id
    with pytest.raises(ValueError):
        label_anomalous_by_life_fraction(df, rul, healthy_frac=0.85)


def test_invalid_healthy_frac_raises():
    df = _make_test_df(engine_id=1, n_observed_cycles=50)
    rul = pd.Series({1: 10}, name="RUL")
    with pytest.raises(ValueError):
        label_anomalous_by_life_fraction(df, rul, healthy_frac=0.0)
    with pytest.raises(ValueError):
        label_anomalous_by_life_fraction(df, rul, healthy_frac=1.0)


# --- compute_engine_total_life ------------------------------------------


def test_compute_engine_total_life():
    df = pd.concat(
        [_make_test_df(1, n_observed_cycles=95), _make_test_df(2, n_observed_cycles=40)],
        ignore_index=True,
    )
    rul = pd.Series({1: 5, 2: 60}, name="RUL")

    total_life = compute_engine_total_life(df, rul)

    assert total_life.loc[1] == 100
    assert total_life.loc[2] == 100


# --- evaluate_detection ------------------------------------------------


def test_evaluate_detection_perfect_predictions():
    # 2 engines, 3 windows each. y_true matches alerts exactly.
    y_true = np.array([False, False, True, False, False, True])
    scores = np.array([0.1, 0.2, 0.9, 0.1, 0.2, 0.9])
    alerts = np.array([False, False, True, False, False, True])
    engine_ids = np.array([1, 1, 1, 2, 2, 2])
    end_cycles = np.array([10, 20, 30, 10, 20, 30])
    total_life = pd.Series({1: 30, 2: 30})

    result = evaluate_detection(y_true, scores, alerts, engine_ids, end_cycles, total_life)

    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)
    assert result.false_alarm_rate == pytest.approx(0.0)
    assert result.detection_rate == pytest.approx(1.0)
    assert result.n_engines == 2


def test_evaluate_detection_missed_engine_excluded_from_mean_lead_time():
    # Engine 1 detected at cycle 30 (failure at 30 -> lead_time=0).
    # Engine 2 never alerts -> missed, excluded from mean_lead_time.
    y_true = np.array([False, False, True, False, False, True])
    scores = np.array([0.1, 0.2, 0.9, 0.1, 0.2, 0.3])
    alerts = np.array([False, False, True, False, False, False])
    engine_ids = np.array([1, 1, 1, 2, 2, 2])
    end_cycles = np.array([10, 20, 30, 10, 20, 30])
    total_life = pd.Series({1: 30, 2: 30})

    result = evaluate_detection(y_true, scores, alerts, engine_ids, end_cycles, total_life)

    assert result.detection_rate == pytest.approx(0.5)
    assert result.mean_lead_time == pytest.approx(0.0)  # only engine 1 counted


def test_evaluate_detection_no_detections_gives_none_mean_lead_time():
    y_true = np.array([True, True])
    scores = np.array([0.1, 0.1])
    alerts = np.array([False, False])
    engine_ids = np.array([1, 1])
    end_cycles = np.array([10, 20])
    total_life = pd.Series({1: 20})

    result = evaluate_detection(y_true, scores, alerts, engine_ids, end_cycles, total_life)

    assert result.detection_rate == pytest.approx(0.0)
    assert result.mean_lead_time is None


def test_evaluate_detection_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        evaluate_detection(
            y_true=np.array([True, False]),
            scores=np.array([0.1]),
            alerts=np.array([True, False]),
            engine_ids=np.array([1, 1]),
            end_cycles=np.array([10, 20]),
            engine_total_life=pd.Series({1: 20}),
        )


def test_evaluate_detection_missing_engine_in_total_life_raises():
    with pytest.raises(ValueError):
        evaluate_detection(
            y_true=np.array([True]),
            scores=np.array([0.9]),
            alerts=np.array([True]),
            engine_ids=np.array([1]),
            end_cycles=np.array([10]),
            engine_total_life=pd.Series({2: 20}),  # wrong engine
        )