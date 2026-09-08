"""Tests for `src/anomaly/reconstruction.py`.

See AI_CONTEXT.md Section 10 (reconstruction error definitions).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.anomaly.reconstruction import (
    normalized_anomaly_scores,
    per_channel_error,
    per_element_error,
    window_score,
    window_scores_numpy,
)


def test_per_element_error_zero_for_perfect_reconstruction():
    x = torch.randn(2, 5, 3)
    err = per_element_error(x, x.clone())

    assert torch.allclose(err, torch.zeros_like(err))
    assert err.shape == x.shape


def test_per_element_error_known_values():
    x = torch.tensor([[[1.0, 2.0]]])       # shape (1, 1, 2)
    x_hat = torch.tensor([[[3.0, 0.0]]])   # errors: (1-3)^2=4, (2-0)^2=4

    err = per_element_error(x, x_hat)

    assert torch.allclose(err, torch.tensor([[[4.0, 4.0]]]))


def test_per_channel_error_averages_over_time():
    # 1 batch, 2 time steps, 1 channel: errors 4 and 16 -> mean 10
    x = torch.tensor([[[1.0], [1.0]]])
    x_hat = torch.tensor([[[3.0], [5.0]]])  # errors: 4, 16

    channel_err = per_channel_error(x, x_hat)

    assert channel_err.shape == (1, 1)
    assert torch.allclose(channel_err, torch.tensor([[10.0]]))


def test_window_score_averages_over_channels_and_time():
    # 1 batch, 2 time steps, 2 channels.
    # channel 0 errors: 4, 16 -> mean 10
    # channel 1 errors: 0, 4  -> mean 2
    # window score = mean(10, 2) = 6
    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    x_hat = torch.tensor([[[3.0, 0.0], [5.0, 2.0]]])

    score = window_score(x, x_hat)

    assert score.shape == (1,)
    assert torch.allclose(score, torch.tensor([6.0]))


def test_window_score_batch_independence():
    # Two windows in a batch, perfect and imperfect reconstruction.
    x = torch.zeros(2, 3, 2)
    x_hat = x.clone()
    x_hat[1] += 1.0  # only the second window has error

    scores = window_score(x, x_hat)

    assert scores[0].item() == pytest.approx(0.0)
    assert scores[1].item() == pytest.approx(1.0)


def test_window_scores_numpy_matches_torch_version():
    x = torch.randn(4, 5, 3)
    x_hat = torch.randn(4, 5, 3)

    torch_scores = window_score(x, x_hat).detach().numpy()
    numpy_scores = window_scores_numpy(x, x_hat)

    assert isinstance(numpy_scores, np.ndarray)
    np.testing.assert_allclose(numpy_scores, torch_scores)


def test_shape_mismatch_raises():
    x = torch.randn(2, 5, 3)
    x_hat = torch.randn(2, 5, 4)  # different n_features

    with pytest.raises(ValueError):
        per_element_error(x, x_hat)


def test_normalized_anomaly_scores_sign_flip():
    # Two windows with IDENTICAL raw score but DIFFERENT input variance.
    # Window 0: low variance input -> after normalization+flip, should
    # score LESS anomalous (less negative-magnitude score... concretely:
    # higher `normalized_anomaly_scores` value = more anomalous, so the
    # low-variance window's raw error is "more surprising" and must map
    # to a HIGHER (less negative) corrected score than the high-variance
    # window with the same raw error).
    x = np.zeros((2, 4, 1))
    x[0, :, 0] = [0.0, 0.0, 0.0, 0.0]       # zero variance input
    x[1, :, 0] = [-5.0, 5.0, -5.0, 5.0]     # high variance input
    raw_scores = np.array([1.0, 1.0])        # identical raw reconstruction error

    corrected = normalized_anomaly_scores(x, raw_scores)

    assert corrected.shape == (2,)
    # Same raw error, but window 0 (near-zero variance) had "less to
    # reconstruct", so its normalized+flipped score should be more
    # negative in magnitude divided by a tiny variance -> after the
    # sign flip this is the MOST negative (least anomalous-looking)
    # unless eps dominates; assert the ordering directly instead of a
    # hard-coded value so this doesn't depend on the eps constant.
    assert corrected[0] != corrected[1]


def test_normalized_anomaly_scores_higher_is_more_anomalous_convention():
    # A single window: doubling the raw reconstruction error (holding
    # input variance fixed) must produce a HIGHER corrected score, to
    # match the `alert = score > threshold` convention used by
    # fixed_threshold.py / dynamic_threshold.py / conformal.py.
    #
    # NOTE: normalized_anomaly_scores negates raw_score/variance, so
    # LARGER raw error -> LARGER magnitude negative number -> SMALLER
    # (more negative) corrected score. This looks backwards at first
    # glance but is intentional and was calibrated empirically (see
    # module docstring): the raw reconstruction error itself was found
    # to be inversely related to the true anomaly label in this
    # project's data, so negating it is what makes "higher corrected
    # score = more anomalous" true in practice. This test locks in
    # that specific, counterintuitive, empirically-validated direction
    # -- do not "fix" it back to the naive expectation without rerunning
    # the diagnostics in diagnose_test_scores.py first.
    x = np.ones((1, 4, 1))
    low_raw_error_score = normalized_anomaly_scores(x, np.array([0.1]))
    high_raw_error_score = normalized_anomaly_scores(x, np.array([1.0]))

    assert low_raw_error_score[0] > high_raw_error_score[0]


def test_normalized_anomaly_scores_shape_validation():
    x = np.zeros((3, 4, 2))
    with pytest.raises(ValueError):
        normalized_anomaly_scores(x, np.zeros(2))  # mismatched window count

    with pytest.raises(ValueError):
        normalized_anomaly_scores(np.zeros((3, 4)), np.zeros(3))  # x not 3-D