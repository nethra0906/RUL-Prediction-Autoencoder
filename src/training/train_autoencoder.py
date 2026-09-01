"""Training loop for the (baseline or conditioned) autoencoder.

See AI_CONTEXT.md Section 9.1 (training protocol elements to track)
and Section 19 (`training/train_autoencoder.py` module contract).

Trains on reconstruction MSE only (unsupervised — no labels needed),
per the baseline system definition in Section 8.1. Early stopping
watches validation loss, matching the closest prior-art paper's
protocol (patience-based, no fixed epoch count assumed to be optimal).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed import set_seed


@dataclass
class TrainingHistory:
    """Per-epoch training record, for experiment logging (Section 21).

    Attributes:
        train_losses: Training-set MSE loss per epoch.
        val_losses: Validation-set MSE loss per epoch.
        best_epoch: Index (0-based) of the epoch with the lowest
            validation loss — the epoch `best_state_dict` was captured
            from.
        stopped_early: Whether training halted before `epochs` due to
            the early-stopping patience being exceeded.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = -1
    stopped_early: bool = False


def train_autoencoder(
    model: nn.Module,
    train_X: torch.Tensor,
    val_X: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    early_stopping_patience: int | None = 10,
    train_settings: torch.Tensor | None = None,
    val_settings: torch.Tensor | None = None,
) -> tuple[nn.Module, TrainingHistory]:
    """Train an autoencoder on healthy-cycle windows via reconstruction MSE.

    Args:
        model: An autoencoder module whose `forward(x)` returns
            `(reconstruction, latent)`, matching
            `models.autoencoder.DenseAutoencoder`'s interface.
        train_X: Training input windows, shape
            (n_train, window_size, n_features). Should already be
            restricted to the healthy region and normalized (see
            `data/healthy_region.py`, `data/normalization.py`) before
            being passed here — this function does not filter or
            normalize.
        val_X: Validation input windows, same shape convention as
            `train_X`, from engines disjoint from `train_X`
            (see `data/splits.py`) and likewise healthy-region +
            normalized.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam optimizer learning rate.
        seed: Random seed, applied via `utils.seed.set_seed` before
            training for reproducibility (AI_CONTEXT.md Section 20/21).
            Always pass explicitly from experiment configuration.
        early_stopping_patience: Number of consecutive epochs without
            validation-loss improvement before stopping early. `None`
            disables early stopping (always runs the full `epochs`).

    Returns:
        `(model, history)`: `model`'s weights are set to the
        best-validation-loss checkpoint encountered (not necessarily
        the final epoch's weights). `history` records per-epoch losses
        and which epoch was selected, for experiment logging.

    Raises:
        ValueError: If `epochs`, `batch_size` are not positive
            integers, or `train_X`/`val_X` are empty.
    """
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if train_X.shape[0] == 0:
        raise ValueError("train_X must be non-empty")
    if val_X.shape[0] == 0:
        raise ValueError("val_X must be non-empty")
    if (train_settings is None) != (val_settings is None):
        raise ValueError(
            "train_settings and val_settings must either both be provided or both be None"
        )

    if train_settings is not None:
        if train_settings.shape[0] != train_X.shape[0]:
            raise ValueError(
                "train_settings must have the same number of samples as train_X"
            )

        if val_settings.shape[0] != val_X.shape[0]:
            raise ValueError(
                "val_settings must have the same number of samples as val_X"
            )

    set_seed(seed)

    if train_settings is None:
        train_dataset = TensorDataset(train_X)
    else:
        train_dataset = TensorDataset(train_X, train_settings)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    history = TrainingHistory()
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()

            if train_settings is None:
                batch_x = batch[0]
                recon, _ = model(batch_x)
            else:
                batch_x, batch_settings = batch
                recon, _ = model(batch_x, batch_settings)

            loss = loss_fn(recon, batch_x)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_x.shape[0]
            train_n += batch_x.shape[0]
        train_loss = train_loss_sum / train_n

        model.eval()
        with torch.no_grad():
            if val_settings is None:
                val_recon, _ = model(val_X)
            else:
                val_recon, _ = model(val_X, val_settings)

            val_loss = loss_fn(val_recon, val_X).item()

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            history.stopped_early = True
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history


__all__ = ["TrainingHistory", "train_autoencoder"]
