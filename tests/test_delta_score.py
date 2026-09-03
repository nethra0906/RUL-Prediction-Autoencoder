"""Tests for `src/anomaly/delta_score.py`."""

from __future__ import annotations

import numpy as np
import pytest

from src.anomaly.delta_score import compute_expanding_delta_scores


def test_first_window_per_engine_has_zero_delta():
    scores = np.array([5.0, 1.0, 1.0])
    engine_ids = np.array([1, 2, 2])
    end_cycles = np.array([10, 10, 20])

    delta = compute_expanding_delta_scores(scores, engine_ids, end_cycles)

    # engine 1's only window, engine 2's first window (cycle 10) -> both 0.
    assert delta[0] == pytest.approx(0.0)
    assert delta[1] == pytest.approx(0.0)


def test_expanding_mean_matches_manual_calculation():
    # Single engine, 4 windows: scores 1, 2, 3, 10.
    # delta[0] = 0 (no history)
    # delta[1] = 2 - mean([1]) = 1
    # delta[2] = 3 - mean([1,2]) = 1.5
    # delta[3] = 10 - mean([1,2,3]) = 8
    scores = np.array([1.0, 2.0, 3.0, 10.0])
    engine_ids = np.array([1, 1, 1, 1])
    end_cycles = np.array([10, 20, 30, 40])

    delta = compute_expanding_delta_scores(scores, engine_ids, end_cycles)

    assert delta.tolist() == pytest.approx([0.0, 1.0, 1.5, 8.0])


def test_rolling_baseline_window_limits_history():
    # baseline_window=2: only the last 2 prior scores are averaged.
    scores = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
    engine_ids = np.array([1] * 5)
    end_cycles = np.array([10, 20, 30, 40, 50])

    delta = compute_expanding_delta_scores(scores, engine_ids, end_cycles, baseline_window=2)

    # delta[4] = 10 - mean(scores[2:4]) = 10 - mean([1, 10]) = 10 - 5.5 = 4.5
    assert delta[4] == pytest.approx(4.5)


def test_engines_are_independent():
    scores = np.array([1.0, 100.0, 1.0, 2.0])
    engine_ids = np.array([1, 1, 2, 2])
    end_cycles = np.array([10, 20, 10, 20])

    delta = compute_expanding_delta_scores(scores, engine_ids, end_cycles)

    # engine 1: delta[1] = 100 - 1 = 99
    # engine 2: delta[1] = 2 - 1 = 1  (unaffected by engine 1's huge jump)
    assert delta[1] == pytest.approx(99.0)
    assert delta[3] == pytest.approx(1.0)


def test_out_of_order_end_cycles_are_sorted_correctly():
    # Windows given out of temporal order; function must sort internally.
    scores = np.array([5.0, 1.0, 3.0])
    engine_ids = np.array([1, 1, 1])
    end_cycles = np.array([30, 10, 20])  # true order: 10, 20, 30 -> scores 1, 3, 5

    delta = compute_expanding_delta_scores(scores, engine_ids, end_cycles)

    # delta at cycle 30 (index 0 in input) = 5 - mean([1, 3]) = 3
    assert delta[0] == pytest.approx(3.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_expanding_delta_scores(
            scores=np.array([1.0, 2.0]),
            engine_ids=np.array([1]),
            end_cycles=np.array([10, 20]),
        )


def test_invalid_baseline_window_raises():
    scores = np.array([1.0, 2.0])
    engine_ids = np.array([1, 1])
    end_cycles = np.array([10, 20])
    with pytest.raises(ValueError):
        compute_expanding_delta_scores(scores, engine_ids, end_cycles, baseline_window=0)