from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class RegimeEncoder(nn.Module):
    """Maps the three operational settings to a learned regime embedding.

    Input:
        Tensor of shape (batch, 3), containing:
            setting_1, setting_2, setting_3

    Output:
        Tensor of shape (batch, embedding_dim).
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: Sequence[int] = (16,),
        embedding_dim: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(
                f"input_dim must be a positive integer, got {input_dim!r}"
            )

        if not all(
            isinstance(h, int) and h > 0
            for h in hidden_dims
        ):
            raise ValueError(
                f"hidden_dims must contain positive integers, got {hidden_dims!r}"
            )

        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be a positive integer, got {embedding_dim!r}"
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout!r}"
            )

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.embedding_dim = embedding_dim

        dims = [input_dim, *self.hidden_dims, embedding_dim]

        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            is_last = i == len(dims) - 2

            if not is_last:
                layers.append(nn.ReLU())

                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, settings: torch.Tensor) -> torch.Tensor:
        """Generate a regime embedding from operational settings.

        Args:
            settings:
                Tensor of shape (batch, 3).

        Returns:
            Tensor of shape (batch, embedding_dim).
        """
        if settings.dim() != 2 or settings.shape[1] != self.input_dim:
            raise ValueError(
                f"expected input shape (batch, {self.input_dim}), "
                f"got {tuple(settings.shape)}"
            )

        return self.network(settings)


def build_regime_encoder_from_config(
    config: dict,
) -> RegimeEncoder:
    """Construct a RegimeEncoder from an experiment config."""

    model_cfg = config.get("model", {})
    regime_cfg = model_cfg.get("regime_encoder", {})

    return RegimeEncoder(
        input_dim=regime_cfg.get("input_dim", 3),
        hidden_dims=tuple(
            regime_cfg.get("hidden_dims", (16,))
        ),
        embedding_dim=regime_cfg.get("embedding_dim", 8),
        dropout=regime_cfg.get("dropout", 0.0),
    )


__all__ = [
    "RegimeEncoder",
    "build_regime_encoder_from_config",
]