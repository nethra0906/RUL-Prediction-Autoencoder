"""Tests for `src/data/detrend.py`."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.detrend import detrend_windows


def test_pure_linear_signal_detrends_to_near_zero():
    # 1 window, 10 time steps, 1 channel, perfect linear ramp.
    t = np.arange(10, dtype=float)
    X = t.reshape(1, 10, 1)

    detrended = detrend_windows(X)

    assert np.allclose(detrended, 0.0, atol=1e-10)


def test_constant_channel_detrends_to_zero():
    X = np.full((1, 10, 1), 5.0)

    detrended = detrend_windows(X)

    assert np.allclose(detrended, 0.0, atol=1e-10)


def test_shape_is_preserved():
    X = np.random.randn(4, 15, 6)

    detrended = detrend_windows(X)

    assert detrended.shape == X.shape


def test_linear_plus_noise_reduces_variance_along_trend_direction():
    # A ramp plus small noise: after detrending, variance should be
    # much lower than before (most of the original variance came from
    # the trend, not the noise).
    rng = np.random.default_rng(0)
    t = np.arange(30, dtype=float)
    noise = rng.normal(scale=0.01, size=30)
    signal = t + noise
    X = signal.reshape(1, 30, 1)

    detrended = detrend_windows(X)

    assert detrended.var() < X.var()


def test_multiple_windows_and_channels_independent():
    # Window 0: channel 0 is a ramp, channel 1 is constant.
    # Window 1: channel 0 is constant, channel 1 is a ramp.
    t = np.arange(5, dtype=float)
    X = np.zeros((2, 5, 2))
    X[0, :, 0] = t
    X[0, :, 1] = 3.0
    X[1, :, 0] = 7.0
    X[1, :, 1] = t

    detrended = detrend_windows(X)

    assert np.allclose(detrended[0, :, 0], 0.0, atol=1e-10)  # ramp removed
    assert np.allclose(detrended[0, :, 1], 0.0, atol=1e-10)  # constant removed
    assert np.allclose(detrended[1, :, 0], 0.0, atol=1e-10)
    assert np.allclose(detrended[1, :, 1], 0.0, atol=1e-10)


def test_non_3d_input_raises():
    with pytest.raises(ValueError):
        detrend_windows(np.zeros((5, 10)))


def test_window_size_below_2_raises():
    with pytest.raises(ValueError):
        detrend_windows(np.zeros((3, 1, 4)))