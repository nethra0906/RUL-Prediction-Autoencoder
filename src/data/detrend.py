"""Per-window linear detrending, applied along the time axis.

See project diagnostic (30 Aug 2026): a dense autoencoder was found to
reconstruct smooth monotonic degradation trends MORE easily than
stationary healthy noise, producing lower reconstruction error for
degrading windows than healthy ones — the inverse of the intended
signal (confirmed by a monotonically decreasing mean-score-by-life-
fraction trend spanning the entire healthy region, not just the
anomaly boundary).

Detrending removes each window's own linear trend before the model
ever sees it, so a straight-line drift is never something the model
gets "credit" for reconstructing well. What's left is only the
residual (unexplained-by-a-straight-line) signal. A genuinely
nonlinear degradation pattern leaves curvature a linear fit can't
remove, which should surface as elevated residual/reconstruction error
relative to the (closer-to-linear, closer-to-flat) healthy case.
"""

from __future__ import annotations

import numpy as np


def detrend_windows(X: np.ndarray) -> np.ndarray:
    """Remove a per-window, per-channel linear trend from windowed data.

    For each (window, channel) pair, fits `y = a*t + b` via least
    squares against `t = 0..window_size-1`, then subtracts the fitted
    line, leaving only the residual.

    Args:
        X: Windowed tensor, shape (n_windows, window_size, n_features).
            Typically `WindowedSequences.X` from `data/windows.py`,
            already normalized.

    Returns:
        Detrended tensor, same shape as `X`.

    Raises:
        ValueError: If `X` is not 3-D or `window_size < 2` (a linear
            fit needs at least 2 points).
    """
    if X.ndim != 3:
        raise ValueError(
            f"X must be 3-D (n_windows, window_size, n_features), got shape {X.shape}"
        )
    _, window_size, _ = X.shape
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2 to fit a linear trend, got {window_size}")

    t = np.arange(window_size, dtype=float)
    t_centered = t - t.mean()
    denom = np.sum(t_centered**2)  # scalar; same for every window/channel

    mean_X = X.mean(axis=1, keepdims=True)  # shape (n_windows, 1, n_features)
    # slope[w, f] = sum_t t_centered[t] * X[w, t, f] / denom
    slope = np.einsum("t,wtf->wf", t_centered, X) / denom  # shape (n_windows, n_features)

    trend = mean_X + slope[:, None, :] * t_centered[None, :, None]
    return X - trend


__all__ = ["detrend_windows"]