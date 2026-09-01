import pytest
import torch

from src.models.regime_encoder import RegimeEncoder


def test_regime_encoder_output_shape():
    model = RegimeEncoder(
        input_dim=3,
        hidden_dims=(16,),
        embedding_dim=8,
    )

    x = torch.randn(10, 3)
    y = model(x)

    assert y.shape == (10, 8)


def test_regime_encoder_rejects_wrong_shape():
    model = RegimeEncoder()

    with pytest.raises(ValueError):
        model(torch.randn(10, 4))


def test_regime_encoder_is_deterministic_in_eval_mode():
    model = RegimeEncoder()
    model.eval()

    x = torch.randn(10, 3)

    y1 = model(x)
    y2 = model(x)

    assert torch.allclose(y1, y2)


def test_regime_encoder_custom_dimensions():
    model = RegimeEncoder(
        input_dim=3,
        hidden_dims=(32, 16),
        embedding_dim=12,
    )

    x = torch.randn(7, 3)
    y = model(x)

    assert y.shape == (7, 12)