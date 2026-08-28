"""Asymmetric NASA (PHM08-style) scoring function for RUL predictions.

See AI_CONTEXT.md Section 14 ("NASA scoring function") and Section G6
(symmetric losses under-penalize late predictions, which are more
operationally dangerous than early ones).

This module is intentionally isolated and unit-tested on its own (per
the `evaluation/nasa_score.py` contract in AI_CONTEXT.md Section 19),
separate from `rul_metrics.py`'s symmetric MAE/RMSE.

Definition (Saxena et al., the standard PHM08 / C-MAPSS scoring
function):

    d_i = RUL_pred_i - RUL_true_i

    s_i = exp(-d_i / 13) - 1   if d_i < 0   (early prediction)
    s_i = exp( d_i / 10) - 1   if d_i >= 0  (late prediction)

    NASA_score = sum_i s_i

A late prediction (predicted RUL greater than the true RUL — i.e. the
model is over-optimistic about remaining life and would delay
maintenance past the real failure point) is penalized with the steeper
`/10` exponent, while an early prediction uses the gentler `/13`
exponent. The score is not an average: as with the original
formulation, larger evaluation sets naturally produce larger totals, so
NASA scores should only be compared across runs evaluated on the same
set of engines.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def nasa_score(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    early_denom: float = 13.0,
    late_denom: float = 10.0,
) -> float:
    """Compute the asymmetric NASA/PHM08 RUL scoring function.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values. Must be the same length as `y_true`.
        early_denom: Denominator for early predictions (d < 0). Defaults
            to the standard PHM08 value of 13.
        late_denom: Denominator for late predictions (d >= 0). Defaults
            to the standard PHM08 value of 10. Must be < `early_denom`
            (or at least small enough) to keep the intended asymmetry —
            this is only asserted, not silently corrected, so a
            deliberate override is never overridden back.

    Returns:
        Total (summed, not averaged) NASA score across all samples.
        Lower is better; a perfect predictor scores 0.

    Raises:
        ValueError: If inputs are empty, of mismatched length, or the
            denominators are non-positive.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.size == 0:
        raise ValueError("y_true/y_pred must be non-empty")
    if early_denom <= 0 or late_denom <= 0:
        raise ValueError("early_denom and late_denom must be positive")

    d = yp - yt
    scores = np.where(d < 0, np.exp(-d / early_denom) - 1, np.exp(d / late_denom) - 1)
    return float(np.sum(scores))


def nasa_score_per_sample(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    early_denom: float = 13.0,
    late_denom: float = 10.0,
) -> np.ndarray:
    """Per-sample NASA score contributions (useful for early/late error
    analysis plots — see AI_CONTEXT.md Section 23, RUL visualization).

    Same arguments and asymmetry as `nasa_score`, but returns the
    per-element array `s_i` instead of the sum.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.size == 0:
        raise ValueError("y_true/y_pred must be non-empty")
    if early_denom <= 0 or late_denom <= 0:
        raise ValueError("early_denom and late_denom must be positive")

    d = yp - yt
    return np.where(d < 0, np.exp(-d / early_denom) - 1, np.exp(d / late_denom) - 1)


__all__ = ["nasa_score", "nasa_score_per_sample"]
