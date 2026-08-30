"""Baseline (vanilla) autoencoder architecture.

See AI_CONTEXT.md Section 9.1 (baseline autoencoder guidance) and
Section 19 (`models/autoencoder.py` module contract).

This is a simple dense (fully-connected) autoencoder operating on
flattened sliding windows — the "vanilla AE" baseline referenced in
the Day 5 task and Section 8.1 (baseline system). It is NOT the LSTM
architecture used by the closest prior-art paper (arXiv 2601.10269);
that remains a possible later experiment variant, not this baseline.

Per AI_CONTEXT.md Section 9.1, layer sizes are an experiment variable
and must never be hard-coded as "the" architecture without evidence —
`hidden_dims` and `latent_dim` are always passed in from configuration.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    """Fully-connected autoencoder over a flattened sensor window.

    Architecture (see AI_CONTEXT.md Section 9.1):

        Input window (window_size, n_features)
            -> flatten
            -> Dense encoder (hidden_dims, shrinking) -> latent (latent_dim)
            -> Dense decoder (mirror of encoder)
            -> reshape back to (window_size, n_features)

    The decoder's final layer has no activation (linear output), since
    inputs are expected to already be normalized (see
    `data/normalization.py`) rather than bounded in [0, 1] — a sigmoid
    output would be wrong for z-scored or condition-normalized sensor
    values.
    """

    def __init__(
        self,
        window_size: int,
        n_features: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            window_size: Number of cycles per input window (matches
                `data/windows.py`'s `window_size`).
            n_features: Number of feature channels per cycle (e.g. 24
                for 3 settings + 21 sensors, or fewer if a feature
                subset was selected — matches the last dim of
                `WindowedSequences.X`).
            latent_dim: Bottleneck dimensionality. Always pass this
                explicitly from experiment configuration
                (AI_CONTEXT.md Section 20) — e.g. `configs/base.yaml`
                `model.latent_dim`.
            hidden_dims: Sizes of the encoder's hidden layers, in
                order from input side to bottleneck side. The decoder
                mirrors this in reverse. Defaults to (64, 32) as a
                starting point only — not an established project
                value; log whatever is actually used per experiment.
            dropout: Dropout probability applied after each hidden
                encoder/decoder layer (not after the bottleneck or
                final output layer). 0.0 disables dropout.

        Raises:
            ValueError: If any dimension argument is not a positive
                integer, or `dropout` is not in [0, 1).
        """
        super().__init__()

        for name, value in [
            ("window_size", window_size),
            ("n_features", n_features),
            ("latent_dim", latent_dim),
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not all(isinstance(h, int) and h > 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims must all be positive integers, got {hidden_dims!r}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

        self.window_size = window_size
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_dims = tuple(hidden_dims)
        self.input_dim = window_size * n_features

        self.encoder = self._build_mlp(
            dims=[self.input_dim, *self.hidden_dims, latent_dim],
            dropout=dropout,
            final_activation=False,
        )
        self.decoder = self._build_mlp(
            dims=[latent_dim, *reversed(self.hidden_dims), self.input_dim],
            dropout=dropout,
            final_activation=False,
        )

    @staticmethod
    def _build_mlp(dims: Sequence[int], dropout: float, final_activation: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        n_layers = len(dims) - 1
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == n_layers - 1
            if not is_last or final_activation:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a batch of windows.

        Args:
            x: Input tensor of shape (batch, window_size, n_features).

        Returns:
            (reconstruction, latent):
                reconstruction: shape (batch, window_size, n_features),
                    same shape as `x`.
                latent: shape (batch, latent_dim) — the bottleneck
                    representation, exposed for potential downstream
                    use (e.g. a future joint RUL head per Section 8.4;
                    not used by the baseline detector itself).

        Raises:
            ValueError: If `x`'s last two dimensions don't match
                `(window_size, n_features)`.
        """
        if x.dim() != 3 or x.shape[1:] != (self.window_size, self.n_features):
            raise ValueError(
                f"expected input shape (batch, {self.window_size}, {self.n_features}), "
                f"got {tuple(x.shape)}"
            )

        batch_size = x.shape[0]
        flat = x.reshape(batch_size, self.input_dim)

        latent = self.encoder(flat)
        recon_flat = self.decoder(latent)
        reconstruction = recon_flat.reshape(batch_size, self.window_size, self.n_features)

        return reconstruction, latent


def build_autoencoder_from_config(
    config: dict,
    window_size: int,
    n_features: int,
) -> DenseAutoencoder:
    """Construct a `DenseAutoencoder` from an experiment config dict.

    Reads `config["model"]["latent_dim"]` (required) and optionally
    `config["model"]["hidden_dims"]` / `config["model"]["dropout"]`
    (fall back to `DenseAutoencoder` defaults if absent). Matches the
    `model:` block in `configs/base.yaml`.

    Args:
        config: Parsed experiment config (e.g. loaded from
            `configs/base.yaml` merged with dataset/experiment
            overrides).
        window_size: Window size actually used to build the input
            tensors (from `data/windows.py`), passed explicitly rather
            than re-read from config to guarantee the model always
            matches the data it will see.
        n_features: Number of feature channels actually used, for the
            same reason.

    Returns:
        A configured, untrained `DenseAutoencoder`.

    Raises:
        KeyError: If `config["model"]["latent_dim"]` is missing.
    """
    model_cfg = config.get("model", {})
    if "latent_dim" not in model_cfg:
        raise KeyError("config['model']['latent_dim'] is required to build the autoencoder")

    kwargs = {"window_size": window_size, "n_features": n_features, "latent_dim": model_cfg["latent_dim"]}
    if "hidden_dims" in model_cfg:
        kwargs["hidden_dims"] = tuple(model_cfg["hidden_dims"])
    if "dropout" in model_cfg:
        kwargs["dropout"] = model_cfg["dropout"]

    return DenseAutoencoder(**kwargs)


__all__ = ["DenseAutoencoder", "build_autoencoder_from_config"]