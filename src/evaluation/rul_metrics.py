"""RUL regression metrics: MAE and RMSE.

See AI_CONTEXT.md Section 14 (RUL evaluation) and Section 19
(`evaluation/rul_metrics.py` contract).

These are symmetric error metrics — they treat early and late RUL errors
identically. The asymmetric, operationally-weighted metric lives in
`evaluation/nasa_score.py` (see AI_CONTEXT.md Section G6 for why the two
are kept separate).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """MAE = mean(|y_true - y_pred|).

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values. Must be the same length as `y_true`.

    Returns:
        Mean absolute error as a float.

    Raises:
        ValueError: If inputs are empty or of mismatched length.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.size == 0:
        raise ValueError("y_true/y_pred must be non-empty")
    return float(np.mean(np.abs(yt - yp)))


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """RMSE = sqrt(mean((y_true - y_pred)^2)).

    RMSE penalizes large deviations more heavily than MAE (see
    AI_CONTEXT.md Section 14).

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values. Must be the same length as `y_true`.

    Returns:
        Root mean squared error as a float.

    Raises:
        ValueError: If inputs are empty or of mismatched length.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.size == 0:
        raise ValueError("y_true/y_pred must be non-empty")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


__all__ = ["mean_absolute_error", "root_mean_squared_error"]
