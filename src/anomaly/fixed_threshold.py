"""Baseline fixed reconstruction-error threshold detector.

See AI_CONTEXT.md Section 11.1 (fixed baseline) and Section 19
(`anomaly/fixed_threshold.py` module contract).

Definition (Section 11.1):

    threshold = mean(scores) + lambda * std(scores)
    alert = reconstruction_error > threshold

The threshold must be fit on healthy training/calibration data only,
never on test data (AI_CONTEXT.md Section 17 Rule 3: "Never choose a
threshold because it gives the best test-set F1"). This module
enforces that separation structurally: `fit_fixed_threshold` only ever
sees the scores you pass it, and `apply_fixed_threshold` takes an
already-fitted threshold as a plain float — there is no code path here
that can accidentally fit on the same scores being evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedThreshold:
    """A fitted fixed threshold, with the statistics used to derive it.

    Attributes:
        value: The decision threshold itself (mean + lambda * std).
        mean: Mean of the healthy scores the threshold was fit on.
        std: Standard deviation of those scores.
        lambda_: The multiplier used (AI_CONTEXT.md Section 11.1's
            "tunable hyperparameter", e.g. 2.5 in the closest prior-art
            paper — not assumed here, always passed explicitly).
    """

    value: float
    mean: float
    std: float
    lambda_: float


def fit_fixed_threshold(healthy_scores: np.ndarray, lambda_: float) -> FixedThreshold:
    """Fit threshold = mean(healthy_scores) + lambda_ * std(healthy_scores).

    Args:
        healthy_scores: 1-D array of window-level reconstruction-error
            scores (e.g. from `anomaly.reconstruction.window_scores_numpy`)
            computed on healthy data ONLY — training or a held-out
            healthy calibration split, never test data (AI_CONTEXT.md
            Section 17 Rule 3).
        lambda_: Multiplier controlling the false-alarm / missed-
            detection trade-off. Must be passed explicitly from
            experiment configuration (AI_CONTEXT.md Section 20) —
            never hard-coded here. Must be non-negative.

    Returns:
        A `FixedThreshold` capturing the fitted value and the
        statistics it was derived from (useful for logging per
        AI_CONTEXT.md Section 21 experiment tracking).

    Raises:
        ValueError: If `healthy_scores` is empty or not 1-D, or if
            `lambda_` is negative.
    """
    scores = np.asarray(healthy_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"healthy_scores must be 1-D, got shape {scores.shape}")
    if scores.size == 0:
        raise ValueError("healthy_scores must be non-empty")
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be non-negative, got {lambda_!r}")

    mean = float(scores.mean())
    std = float(scores.std())
    value = mean + lambda_ * std

    return FixedThreshold(value=value, mean=mean, std=std, lambda_=lambda_)


def apply_fixed_threshold(scores: np.ndarray, threshold: FixedThreshold | float) -> np.ndarray:
    """Apply an already-fitted threshold to produce binary alerts.

    Args:
        scores: 1-D array of window-level reconstruction-error scores
            to evaluate (e.g. from validation or test windows).
        threshold: Either a `FixedThreshold` (from `fit_fixed_threshold`)
            or a raw float value. Accepting a float too keeps this
            function usable for a manually-specified threshold without
            forcing callers through `fit_fixed_threshold`.

    Returns:
        Boolean array, same shape as `scores`: True where `scores`
        exceeds the threshold (an alert).

    Raises:
        ValueError: If `scores` is not 1-D.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")

    threshold_value = threshold.value if isinstance(threshold, FixedThreshold) else float(threshold)
    return scores > threshold_value


__all__ = ["FixedThreshold", "fit_fixed_threshold", "apply_fixed_threshold"]
