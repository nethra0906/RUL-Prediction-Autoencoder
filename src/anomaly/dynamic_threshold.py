"""Heuristic dynamic reconstruction-error threshold.

See AI_CONTEXT.md Section 18/19 for this module's contract.

The threshold is computed from a rolling window of reconstruction
scores:

    threshold_t = rolling_mean_t + lambda_ * rolling_std_t

The threshold at each position is based only on scores available up
to that position. This avoids using future scores when producing a
decision for the current window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicThreshold:
    """Configuration for a rolling reconstruction-error threshold."""

    window_size: int
    lambda_: float


def fit_dynamic_threshold(
    scores: np.ndarray,
    window_size: int,
    lambda_: float,
) -> np.ndarray:
    """Compute a rolling dynamic threshold from reconstruction scores.

    For each score position, the threshold is calculated from the
    current score and the preceding ``window_size - 1`` scores.

    Args:
        scores: 1-D reconstruction-error scores ordered by time.
        window_size: Number of recent scores used to calculate the
            rolling statistics. Must be positive.
        lambda_: Non-negative multiplier applied to the rolling
            standard deviation.

    Returns:
        A 1-D array containing one threshold for every input score.

    Raises:
        ValueError: If scores are not 1-D or are empty, window_size is
            not positive, or lambda_ is negative.
    """
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1:
        raise ValueError(
            f"scores must be 1-D, got shape {scores.shape}"
        )

    if scores.size == 0:
        raise ValueError("scores must be non-empty")

    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(
            f"window_size must be a positive integer, got {window_size!r}"
        )

    if lambda_ < 0:
        raise ValueError(
            f"lambda_ must be non-negative, got {lambda_!r}"
        )

    thresholds = np.empty_like(scores, dtype=float)

    for i in range(scores.size):
        start = max(0, i - window_size + 1)
        history = scores[start : i + 1]

        mean = history.mean()
        std = history.std()

        thresholds[i] = mean + lambda_ * std

    return thresholds


def apply_dynamic_threshold(
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Apply dynamic thresholds to reconstruction-error scores.

    An alert is raised when the reconstruction score is strictly
    greater than its corresponding threshold.

    Args:
        scores: 1-D reconstruction-error scores.
        thresholds: 1-D threshold array with the same shape as scores.

    Returns:
        Boolean alert array.
    """
    scores = np.asarray(scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if scores.ndim != 1:
        raise ValueError(
            f"scores must be 1-D, got shape {scores.shape}"
        )

    if thresholds.ndim != 1:
        raise ValueError(
            f"thresholds must be 1-D, got shape {thresholds.shape}"
        )

    if scores.shape != thresholds.shape:
        raise ValueError(
            "scores and thresholds must have the same shape, "
            f"got {scores.shape} and {thresholds.shape}"
        )

    return scores > thresholds


__all__ = [
    "DynamicThreshold",
    "fit_dynamic_threshold",
    "apply_dynamic_threshold",
]