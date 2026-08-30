"""Total / per-time-step / per-channel reconstruction error.

See AI_CONTEXT.md Section 10 (reconstruction error) and Section 19
(`anomaly/reconstruction.py` module contract).

Definitions (Section 10), for input x and reconstruction x_hat:

    e = (x - x_hat)^2                          per-element squared error
    E_j = mean_t [(x_t,j - x_hat_t,j)^2]        per-channel error (over time)
    E_total = mean_j E_j                        scalar window-level score

The channel-wise error is preserved (not immediately reduced to a
scalar) so that `anomaly/attribution.py` can later rank per-sensor
contributions from the same computation — see Section 10: "For sensor
attribution, preserve the channel-wise error before reducing it to a
scalar."
"""

from __future__ import annotations

import numpy as np
import torch


def per_element_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Squared error at every (window, time step, channel) position.

    Args:
        x: Original input, shape (batch, window_size, n_features).
        x_hat: Reconstruction, same shape as `x`.

    Returns:
        Squared error tensor, same shape as `x`.

    Raises:
        ValueError: If `x` and `x_hat` shapes don't match.
    """
    if x.shape != x_hat.shape:
        raise ValueError(f"x and x_hat must have the same shape, got {tuple(x.shape)} vs {tuple(x_hat.shape)}")
    return (x - x_hat) ** 2


def per_channel_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Per-channel reconstruction error, averaged over the time axis.

    E_j = mean_t [(x_t,j - x_hat_t,j)^2]  (AI_CONTEXT.md Section 10)

    This is the array `anomaly/attribution.py` should consume directly
    to rank per-sensor contributions for a window — do not re-derive
    it from `window_score` (which has already collapsed the channel
    axis).

    Args:
        x: Original input, shape (batch, window_size, n_features).
        x_hat: Reconstruction, same shape as `x`.

    Returns:
        Per-channel error, shape (batch, n_features).
    """
    sq_err = per_element_error(x, x_hat)
    return sq_err.mean(dim=1)  # average over the time-step axis


def window_score(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Scalar anomaly score per window: E_total = mean_j E_j.

    This is the single number a fixed/dynamic/conformal threshold
    compares against (AI_CONTEXT.md Section 11). Computed as the mean
    of `per_channel_error`, per the reduction defined in Section 10 —
    reused consistently rather than re-derived ad hoc elsewhere.

    Args:
        x: Original input, shape (batch, window_size, n_features).
        x_hat: Reconstruction, same shape as `x`.

    Returns:
        Scalar score per window, shape (batch,).
    """
    channel_err = per_channel_error(x, x_hat)
    return channel_err.mean(dim=1)  # average over the channel axis


def window_scores_numpy(x: torch.Tensor, x_hat: torch.Tensor) -> np.ndarray:
    """Convenience wrapper: `window_score`, detached to a numpy array.

    Useful when the result feeds non-torch downstream code (e.g.
    `anomaly/fixed_threshold.py`, sklearn-style metrics), so callers
    don't need to repeat the `.detach().cpu().numpy()` boilerplate at
    every call site.
    """
    return window_score(x, x_hat).detach().cpu().numpy()


__all__ = [
    "per_element_error",
    "per_channel_error",
    "window_score",
    "window_scores_numpy",
]
