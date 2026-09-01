from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from src.models.autoencoder import DenseAutoencoder
from src.models.regime_encoder import RegimeEncoder


class RegimeConditionedAutoencoder(nn.Module):
    """Autoencoder conditioned on the operating-condition representation.

    The model combines:
        1. A sensor-window representation from the vanilla AE encoder.
        2. A learned regime embedding from the three operational settings.

    The two representations are concatenated and projected into the
    final latent representation before being passed to the decoder.

    Inputs:
        x:
            Sensor window of shape
            (batch, window_size, n_features).

        settings:
            Operational settings of shape (batch, 3):
            [setting_1, setting_2, setting_3].

    Outputs:
        reconstruction:
            Reconstructed sensor window with the same shape as x.

        latent:
            Conditioned latent representation.
    """

    def __init__(
        self,
        window_size: int,
        n_features: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        regime_hidden_dims: Sequence[int] = (16,),
        regime_embedding_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.n_features = n_features
        self.latent_dim = latent_dim

        # Sensor-side encoder.
        self.sensor_autoencoder = DenseAutoencoder(
            window_size=window_size,
            n_features=n_features,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Operating-condition encoder.
        self.regime_encoder = RegimeEncoder(
            input_dim=3,
            hidden_dims=regime_hidden_dims,
            embedding_dim=regime_embedding_dim,
            dropout=dropout,
        )

        # Combine sensor latent + regime embedding.
        self.conditioning_layer = nn.Sequential(
            nn.Linear(
                latent_dim + regime_embedding_dim,
                latent_dim,
            ),
            nn.ReLU(),
        )

        # Reuse the decoder architecture from the vanilla AE.
        self.decoder = self.sensor_autoencoder.decoder

    def forward(
        self,
        x: torch.Tensor,
        settings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a sensor window using regime conditioning.

        Args:
            x:
                Tensor of shape
                (batch, window_size, n_features).

            settings:
                Tensor of shape (batch, 3).

        Returns:
            reconstruction:
                Tensor with shape identical to x.

            latent:
                Conditioned latent tensor of shape
                (batch, latent_dim).
        """

        if x.dim() != 3 or x.shape[1:] != (
            self.window_size,
            self.n_features,
        ):
            raise ValueError(
                f"expected x shape "
                f"(batch, {self.window_size}, {self.n_features}), "
                f"got {tuple(x.shape)}"
            )

        if settings.dim() != 2 or settings.shape[1] != 3:
            raise ValueError(
                f"expected settings shape (batch, 3), "
                f"got {tuple(settings.shape)}"
            )

        if x.shape[0] != settings.shape[0]:
            raise ValueError(
                "x and settings must have the same batch size, "
                f"got {x.shape[0]} and {settings.shape[0]}"
            )

        # Extract sensor representation.
        batch_size = x.shape[0]
        flat = x.reshape(batch_size, self.sensor_autoencoder.input_dim)
        sensor_latent = self.sensor_autoencoder.encoder(flat)

        # Extract regime representation.
        regime_embedding = self.regime_encoder(settings)

        # Combine both representations.
        combined = torch.cat(
            [sensor_latent, regime_embedding],
            dim=1,
        )

        latent = self.conditioning_layer(combined)

        # Decode conditioned latent representation.
        recon_flat = self.decoder(latent)

        reconstruction = recon_flat.reshape(
            batch_size,
            self.window_size,
            self.n_features,
        )

        return reconstruction, latent


def build_conditioned_autoencoder_from_config(
    config: dict,
    window_size: int,
    n_features: int,
) -> RegimeConditionedAutoencoder:
    """Construct a regime-conditioned autoencoder from config."""

    model_cfg = config.get("model", {})

    if "latent_dim" not in model_cfg:
        raise KeyError(
            "config['model']['latent_dim'] is required"
        )

    regime_cfg = model_cfg.get("regime_encoder", {})

    hidden_dims = tuple(
        model_cfg.get("hidden_dims", (64, 32))
    )

    regime_hidden_dims = tuple(
        regime_cfg.get("hidden_dims", (16,))
    )

    return RegimeConditionedAutoencoder(
        window_size=window_size,
        n_features=n_features,
        latent_dim=model_cfg["latent_dim"],
        hidden_dims=hidden_dims,
        regime_hidden_dims=regime_hidden_dims,
        regime_embedding_dim=regime_cfg.get(
            "embedding_dim",
            8,
        ),
        dropout=model_cfg.get("dropout", 0.0),
    )


__all__ = [
    "RegimeConditionedAutoencoder",
    "build_conditioned_autoencoder_from_config",
]