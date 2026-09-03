"""Per-engine expanding/rolling delta scoring.

Reframes each window's anomaly score relative to that SAME engine's
own recent history, rather than as an absolute value. Rationale:
absolute reconstruction error can sit at a low, stable baseline for a
given engine and still carry an early-warning signal in its RATE OF
CHANGE, which a fixed absolute threshold may not catch cleanly.

Causal by construction (see `compute_expanding_delta_scores`): a
window's delta only ever depends on that same engine's STRICTLY
EARLIER windows, never later ones or other engines — a property that
matters both to avoid look-ahead leakage and because this needs to be
usable in an online/deployment setting where future data isn't
available yet.
"""

from __future__ import annotations

import numpy as np


def compute_expanding_delta_scores(
    scores: np.ndarray,
    engine_ids: np.ndarray,
    end_cycles: np.ndarray,
    baseline_window: int | None = None,
) -> np.ndarray:
    """Delta = score - baseline, where baseline is that engine's own
    prior-window history.

    For each engine (sorted by `end_cycles`), window `i`'s baseline is
    the mean of that SAME engine's scores at windows `0..i-1` (an
    expanding mean from the start of the observed trajectory), or —
    if `baseline_window` is given — the mean of only the last
    `baseline_window` prior windows (a rolling baseline instead of an
    ever-growing one).

    A window's own first observation has no prior history, so its
    delta is defined as 0.0 (not NaN) — there's nothing to compare
    against yet, and 0 correctly signals "not yet flaggable" rather
    than propagating a missing value downstream.

    Args:
        scores: 1-D array of window-level anomaly scores (e.g. from
            `anomaly.reconstruction.window_scores_numpy`, typically
            computed on detrended windows — see `data/detrend.py`).
        engine_ids: Engine ID per window (same length as `scores`).
        end_cycles: End cycle per window (same length as `scores`),
            used to establish each engine's temporal order.
        baseline_window: If None (default), use an expanding mean from
            the start of the engine's observed trajectory. If an int,
            use only the most recent `baseline_window` prior windows
            (must be >= 1).

    Returns:
        1-D array of delta scores, same length and order as `scores`.

    Raises:
        ValueError: If `scores`/`engine_ids`/`end_cycles` lengths
            mismatch, or `baseline_window` is not a positive integer
            when given.
    """
    scores = np.asarray(scores, dtype=float)
    engine_ids = np.asarray(engine_ids)
    end_cycles = np.asarray(end_cycles)

    if not (len(scores) == len(engine_ids) == len(end_cycles)):
        raise ValueError(
            "scores, engine_ids, and end_cycles must have the same length, got "
            f"{len(scores)}, {len(engine_ids)}, {len(end_cycles)}"
        )
    if baseline_window is not None and (
        not isinstance(baseline_window, int) or baseline_window < 1
    ):
        raise ValueError(f"baseline_window must be a positive integer or None, got {baseline_window!r}")

    delta = np.zeros_like(scores)

    for engine_id in np.unique(engine_ids):
        mask = engine_ids == engine_id
        idx = np.where(mask)[0]
        order = idx[np.argsort(end_cycles[idx])]
        engine_scores = scores[order]

        engine_delta = np.zeros_like(engine_scores)
        for i in range(1, len(engine_scores)):  # i=0 stays 0.0 (no history)
            if baseline_window is None:
                history = engine_scores[:i]
            else:
                history = engine_scores[max(0, i - baseline_window) : i]
            engine_delta[i] = engine_scores[i] - history.mean()

        delta[order] = engine_delta

    return delta


__all__ = ["compute_expanding_delta_scores"]