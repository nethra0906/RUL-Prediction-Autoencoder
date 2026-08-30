"""Tests for `src/anomaly/reconstruction.py`.

See AI_CONTEXT.md Section 10 (reconstruction error definitions).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.anomaly.reconstruction import (
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