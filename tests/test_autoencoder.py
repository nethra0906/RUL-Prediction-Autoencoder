"""Tests for `src/models/autoencoder.py`.

See AI_CONTEXT.md Section 9.1 (baseline autoencoder guidance).
"""

from __future__ import annotations

import pytest
import torch

from src.models.autoencoder import DenseAutoencoder, build_autoencoder_from_config


def test_output_shape_matches_input_shape():
    model = DenseAutoencoder(window_size=10, n_features=5, latent_dim=4, hidden_dims=(16, 8))
    x = torch.randn(3, 10, 5)  # batch=3

    recon, latent = model(x)

    assert recon.shape == x.shape
    assert latent.shape == (3, 4)


def test_single_sample_batch_works():
    model = DenseAutoencoder(window_size=10, n_features=5, latent_dim=4)
    x = torch.randn(1, 10, 5)

    recon, latent = model(x)

    assert recon.shape == (1, 10, 5)
    assert latent.shape == (1, 4)


def test_gradients_flow_through_full_model():
    model = DenseAutoencoder(window_size=10, n_features=5, latent_dim=4)
    x = torch.randn(2, 10, 5)

    recon, _ = model(x)
    loss = ((recon - x) ** 2).mean()
    loss.backward()

    # Every parameter should have received a gradient.
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_wrong_input_shape_raises():
    model = DenseAutoencoder(window_size=10, n_features=5, latent_dim=4)
    wrong_window = torch.randn(2, 8, 5)  # window_size mismatch
    wrong_features = torch.randn(2, 10, 3)  # n_features mismatch

    with pytest.raises(ValueError):
        model(wrong_window)
    with pytest.raises(ValueError):
        model(wrong_features)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0, "n_features": 5, "latent_dim": 4},
        {"window_size": 10, "n_features": -1, "latent_dim": 4},
        {"window_size": 10, "n_features": 5, "latent_dim": 0},
        {"window_size": 10, "n_features": 5, "latent_dim": 4, "dropout": 1.0},
    ],
)
def test_invalid_dims_raise(kwargs):
    with pytest.raises(ValueError):
        DenseAutoencoder(**kwargs)


def test_invalid_hidden_dims_raise():
    with pytest.raises(ValueError):
        DenseAutoencoder(window_size=10, n_features=5, latent_dim=4, hidden_dims=(16, -8))


def test_build_from_config_uses_latent_dim():
    config = {"model": {"latent_dim": 8}}
    model = build_autoencoder_from_config(config, window_size=10, n_features=5)

    assert model.latent_dim == 8
    assert model.window_size == 10
    assert model.n_features == 5


def test_build_from_config_uses_optional_hidden_dims_and_dropout():
    config = {"model": {"latent_dim": 8, "hidden_dims": [32, 16], "dropout": 0.1}}
    model = build_autoencoder_from_config(config, window_size=10, n_features=5)

    assert model.hidden_dims == (32, 16)


def test_build_from_config_missing_latent_dim_raises():
    config = {"model": {}}
    with pytest.raises(KeyError):
        build_autoencoder_from_config(config, window_size=10, n_features=5)