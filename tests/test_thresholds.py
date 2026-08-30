"""Tests for `src/anomaly/fixed_threshold.py`.

See AI_CONTEXT.md Section 11.1 (fixed baseline) and Section 17 Rule 3
(thresholds must never be fit on test data — enforced here structurally
by never passing test scores to `fit_fixed_threshold`).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.anomaly.fixed_threshold import (
    FixedThreshold,
    apply_fixed_threshold,
    fit_fixed_threshold,
)


def test_threshold_formula_matches_mean_plus_lambda_std():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fit_fixed_threshold(scores, lambda_=2.0)

    expected_mean = scores.mean()
    expected_std = scores.std()
    expected_value = expected_mean + 2.0 * expected_std

    assert result.mean == pytest.approx(expected_mean)
    assert result.std == pytest.approx(expected_std)
    assert result.value == pytest.approx(expected_value)
    assert result.lambda_ == 2.0


def test_lambda_zero_sets_threshold_to_mean():
    scores = np.array([1.0, 2.0, 3.0])
    result = fit_fixed_threshold(scores, lambda_=0.0)

    assert result.value == pytest.approx(scores.mean())


def test_larger_lambda_gives_higher_threshold():
    scores = np.array([1.0, 2.0, 3.0, 10.0])  # has some spread
    low = fit_fixed_threshold(scores, lambda_=1.0)
    high = fit_fixed_threshold(scores, lambda_=3.0)

    assert high.value > low.value


def test_apply_threshold_flags_scores_above_only():
    threshold = FixedThreshold(value=5.0, mean=3.0, std=1.0, lambda_=2.0)
    scores = np.array([1.0, 5.0, 5.1, 10.0])

    alerts = apply_fixed_threshold(scores, threshold)

    # strictly greater than: 5.0 itself is NOT an alert, 5.1 and 10.0 are.
    assert alerts.tolist() == [False, False, True, True]


def test_apply_threshold_accepts_raw_float():
    scores = np.array([1.0, 5.0, 10.0])
    alerts = apply_fixed_threshold(scores, threshold=4.0)

    assert alerts.tolist() == [False, True, True]


def test_fit_then_apply_end_to_end():
    healthy_scores = np.array([1.0, 1.1, 0.9, 1.05, 0.95])  # tight, low-variance
    test_scores = np.array([1.0, 1.0, 5.0])  # last one is a clear anomaly

    threshold = fit_fixed_threshold(healthy_scores, lambda_=2.5)
    alerts = apply_fixed_threshold(test_scores, threshold)

    assert alerts.tolist() == [False, False, True]


def test_empty_scores_raises():
    with pytest.raises(ValueError):
        fit_fixed_threshold(np.array([]), lambda_=2.0)


def test_negative_lambda_raises():
    with pytest.raises(ValueError):
        fit_fixed_threshold(np.array([1.0, 2.0]), lambda_=-1.0)


def test_non_1d_scores_raise():
    with pytest.raises(ValueError):
        fit_fixed_threshold(np.array([[1.0, 2.0], [3.0, 4.0]]), lambda_=2.0)
    with pytest.raises(ValueError):
        apply_fixed_threshold(np.array([[1.0, 2.0]]), threshold=1.0)