import pytest
import torch

from src.models.conditioned_autoencoder import (
    RegimeConditionedAutoencoder,
)


def make_model():
    return RegimeConditionedAutoencoder(
        window_size=30,
        n_features=24,
        latent_dim=16,
        hidden_dims=(64, 32),
        regime_hidden_dims=(16,),
        regime_embedding_dim=8,
    )


def test_conditioned_autoencoder_output_shape():
    model = make_model()

    x = torch.randn(8, 30, 24)
    settings = torch.randn(8, 3)

    reconstruction, latent = model(x, settings)

    assert reconstruction.shape == x.shape
    assert latent.shape == (8, 16)


def test_conditioned_autoencoder_rejects_wrong_sensor_shape():
    model = make_model()

    x = torch.randn(8, 20, 24)
    settings = torch.randn(8, 3)

    with pytest.raises(ValueError):
        model(x, settings)


def test_conditioned_autoencoder_rejects_wrong_settings_shape():
    model = make_model()

    x = torch.randn(8, 30, 24)
    settings = torch.randn(8, 4)

    with pytest.raises(ValueError):
        model(x, settings)


def test_conditioned_autoencoder_rejects_mismatched_batch_size():
    model = make_model()

    x = torch.randn(8, 30, 24)
    settings = torch.randn(7, 3)

    with pytest.raises(ValueError):
        model(x, settings)


def test_conditioned_autoencoder_backward():
    model = make_model()

    x = torch.randn(4, 30, 24)
    settings = torch.randn(4, 3)

    reconstruction, _ = model(x, settings)

    loss = torch.mean((reconstruction - x) ** 2)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(gradient is not None for gradient in gradients)