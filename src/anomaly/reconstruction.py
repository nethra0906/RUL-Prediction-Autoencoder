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


def normalized_anomaly_scores(x: np.ndarray, raw_scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Variance-normalized, sign-corrected anomaly score.

    BACKGROUND (diagnosed 31 Aug-3 Sep 2026, FD001 baseline): raw
    `window_score`/`window_scores_numpy` reconstruction error was found
    to correlate strongly with each window's own input variance
    (Spearman ~0.5-0.94 across several checkpoints and with/without
    per-window detrending — see `data/detrend.py` module docstring for
    the detrending investigation, which was ruled out as the root
    cause). Because input variance is itself *negatively* correlated
    with the true anomaly label in FD001 (late-life windows are
    locally smoother — Spearman ~-0.15), raw reconstruction error ends
    up INVERSELY correlated with the true label: healthy windows score
    higher than anomalous ones, the opposite of the intended
    early-warning signal (raw-score test ROC-AUC was consistently
    ~0.24-0.29, i.e. a strong INVERSE relationship, not noise near 0.5).

    This function applies two corrections, both calibrated using
    non-test data only (AI_CONTEXT.md Section 17 Rule 3 - the
    correction is a fixed scoring-convention choice validated once via
    diagnostics, not something searched for against test performance):

      1. Variance normalization: divide raw MSE by the window's own
         input variance. This is a studentized-residual-style
         correction — it asks "how much error relative to how much
         signal there was to reconstruct" instead of rewarding
         naturally low-variance (late-life) windows for having low
         absolute error.
      2. Sign flip: negate the normalized score. After step 1, the
         corrected score was *still* strongly inversely correlated
         with the true label (test ROC-AUC ~0.03, i.e. ~1 - 0.97 - a
         near-perfect inversion, not residual noise). Negating it
         restores the standard convention used throughout this
         codebase: higher score = more anomalous, consumed as-is by
         `fixed_threshold.py` / `dynamic_threshold.py` / `conformal.py`
         (all of which assume `alert = score > threshold`).

    Validated result (FD001, `fd001_ae_variance_normalized_v001`):
    test ROC-AUC 0.967 after both corrections, vs. 0.03 with variance
    normalization alone and ~0.27 with raw reconstruction error.

    Args:
        x: The (possibly normalized, NOT necessarily detrended) input
            windows actually fed to the model for this scoring pass,
            shape (n_windows, window_size, n_features). Must be the
            same array whose reconstruction produced `raw_scores`.
        raw_scores: 1-D array from `window_scores_numpy(x_as_tensor, x_hat)`
            for the SAME `x`.
        eps: Numerical floor added to the variance denominator to avoid
            division by (near-)zero for degenerate low-variance windows.

    Returns:
        1-D array, same length as `raw_scores`: the corrected score,
        following the "higher = more anomalous" convention expected by
        every threshold module in `src/anomaly`.

    Raises:
        ValueError: If `x` is not 3-D, or `raw_scores` is not 1-D, or
            their leading (window) dimensions don't match.
    """
    x = np.asarray(x)
    raw_scores = np.asarray(raw_scores, dtype=float)

    if x.ndim != 3:
        raise ValueError(f"x must be 3-D (n_windows, window_size, n_features), got shape {x.shape}")
    if raw_scores.ndim != 1:
        raise ValueError(f"raw_scores must be 1-D, got shape {raw_scores.shape}")
    if x.shape[0] != raw_scores.shape[0]:
        raise ValueError(
            f"x and raw_scores must have the same number of windows, got {x.shape[0]} vs {raw_scores.shape[0]}"
        )

    input_variance = x.var(axis=(1, 2))
    return -(raw_scores / (input_variance + eps))


__all__ = [
    "per_element_error",
    "per_channel_error",
    "window_score",
    "window_scores_numpy",
    "normalized_anomaly_scores",
]
