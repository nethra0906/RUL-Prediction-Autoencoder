"""Heuristic dynamic reconstruction-error threshold.

See AI_CONTEXT.md Section 18/19 for this module's contract.

The threshold is computed from a rolling window of reconstruction
scores:

    threshold_t = rolling_mean_t + lambda_ * rolling_std_t

The threshold at each position is based only on scores STRICTLY BEFORE
that position (never including the current score itself, and never
future scores) -- see `fit_dynamic_threshold` docstring for why the
current point is excluded.

BUG FIX (3 Sep 2026): `fit_dynamic_threshold` operates on a single
pooled 1-D array with no notion of engine boundaries. If called
directly on a test set spanning multiple engines (as in
`run_dynamic_threshold.py`), the rolling window at the start of each
new engine's trajectory still contains the tail-end scores of the
PREVIOUS engine -- including that engine's near-failure high scores --
contaminating the threshold right where a new engine's healthy region
begins. Diagnosed via `test_windows.engine_ids` being grouped/sorted
by engine (confirmed: `[1 1 2 2 2 ... 3 3 3 ...]`), combined with
window-level precision collapsing (~0.01-0.02) despite strong ranking
(ROC-AUC ~0.97) and reasonable event-level detection_rate (~0.7) --
a pattern only explainable by per-window threshold corruption at
engine boundaries, not a genuine scoring problem.

`fit_dynamic_threshold_per_engine` fixes this by computing the rolling
threshold independently within each engine's own trajectory (sorted by
`end_cycles`), mirroring the per-engine isolation pattern already used
in `anomaly/delta_score.py`. Callers evaluating across multiple engines
(i.e. any test-set evaluation) should use
`fit_dynamic_threshold_per_engine`, not `fit_dynamic_threshold`
directly, unless they have already manually isolated a single engine's
scores.
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

    For each score position ``i``, the threshold is calculated from
    the preceding ``window_size`` scores -- STRICTLY BEFORE ``i``,
    never including ``scores[i]`` itself.

    BUG FIX (3 Sep 2026): the threshold at index ``i`` previously
    included ``scores[i]`` in its own rolling mean/std (an inclusive
    slice). This let an anomalously high score drag its own threshold
    upward, making it structurally harder for a genuine anomaly to
    ever cross its own threshold -- producing near-zero precision/
    recall despite strong score separation (high ROC-AUC). The history
    window is now strictly prior to the current index.

    This function assumes ``scores`` is already ordered as a SINGLE
    continuous trajectory (e.g. one engine's scores sorted by cycle).
    For multi-engine arrays, use `fit_dynamic_threshold_per_engine`
    instead -- calling this directly on a pooled multi-engine array
    will let one engine's trailing scores contaminate the next
    engine's threshold at the boundary.

    Args:
        scores: 1-D reconstruction-error scores ordered by time, for a
            SINGLE trajectory (see note above).
        window_size: Number of prior scores used to calculate the
            rolling statistics. Must be positive.
        lambda_: Non-negative multiplier applied to the rolling
            standard deviation.

    Returns:
        A 1-D array containing one threshold for every input score.
        The first score in the array has no prior history; its
        threshold falls back to the score's own value (see
        implementation), which is equivalent to "no alert possible at
        the very first observation" once combined with
        `apply_dynamic_threshold`'s strict `>` comparison.

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
        start = max(0, i - window_size)
        history = scores[start:i]  # strictly BEFORE i -- excludes scores[i]

        if history.size == 0:
            # No prior history yet (first observation in the
            # trajectory): nothing to compare against, so no alert
            # should be possible here. Setting the threshold to the
            # score's own value means `score > threshold` is always
            # False (strict inequality), which is the intended
            # "not yet flaggable" behavior -- consistent with how
            # `delta_score.py` defines delta=0.0 for a window with no
            # prior history.
            thresholds[i] = scores[i]
        else:
            mean = history.mean()
            std = history.std()
            thresholds[i] = mean + lambda_ * std

    return thresholds


def fit_dynamic_threshold_per_engine(
    scores: np.ndarray,
    engine_ids: np.ndarray,
    end_cycles: np.ndarray,
    window_size: int,
    lambda_: float,
) -> np.ndarray:
    """Per-engine rolling dynamic threshold.

    Applies `fit_dynamic_threshold` independently within each engine's
    own trajectory (ordered by `end_cycles`), so that one engine's
    scores never contaminate another engine's rolling threshold. See
    module docstring for the cross-engine contamination bug this
    fixes, and `anomaly/delta_score.py`'s `compute_expanding_delta_scores`
    for the analogous per-engine isolation pattern this mirrors.

    Args:
        scores: 1-D window-level anomaly scores, in ANY order (need
            not be pre-sorted or pre-grouped by engine).
        engine_ids: Engine ID per score, same length as `scores`.
        end_cycles: End cycle per score, same length as `scores`,
            used to establish each engine's temporal order.
        window_size: Passed through to `fit_dynamic_threshold`.
        lambda_: Passed through to `fit_dynamic_threshold`.

    Returns:
        1-D array of thresholds, same length and ORIGINAL order as
        `scores` (i.e. safe to compare elementwise against `scores` or
        pass directly to `apply_dynamic_threshold` without any
        additional re-sorting by the caller).

    Raises:
        ValueError: If `scores`/`engine_ids`/`end_cycles` lengths
            mismatch, or (via `fit_dynamic_threshold`) if any single
            engine's score slice is invalid.
    """
    scores = np.asarray(scores, dtype=float)
    engine_ids = np.asarray(engine_ids)
    end_cycles = np.asarray(end_cycles)

    if not (len(scores) == len(engine_ids) == len(end_cycles)):
        raise ValueError(
            "scores, engine_ids, and end_cycles must have the same length, got "
            f"{len(scores)}, {len(engine_ids)}, {len(end_cycles)}"
        )

    thresholds = np.empty_like(scores)

    for engine_id in np.unique(engine_ids):
        mask = engine_ids == engine_id
        idx = np.where(mask)[0]
        order = idx[np.argsort(end_cycles[idx])]

        engine_scores = scores[order]
        engine_thresholds = fit_dynamic_threshold(
            engine_scores, window_size=window_size, lambda_=lambda_
        )
        thresholds[order] = engine_thresholds

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
        thresholds: 1-D threshold array with the same shape as scores
            (e.g. from `fit_dynamic_threshold` for a single trajectory,
            or `fit_dynamic_threshold_per_engine` for a multi-engine
            array -- both return thresholds in the same order as their
            input scores, so no re-alignment is needed here).

    Returns:
        Boolean alert array.

    Raises:
        ValueError: If `scores`/`thresholds` are not 1-D or shapes
            don't match.
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
    "fit_dynamic_threshold_per_engine",
    "apply_dynamic_threshold",
]