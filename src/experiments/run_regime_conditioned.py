from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.healthy_region import select_healthy_region
from src.data.loaders import load_train
from src.data.schema import SENSOR_COLUMNS, SETTING_COLUMNS
from src.data.normalization import fit_regime_normalizer, transform_by_regime
from src.data.regimes import assign_regimes, fit_regime_model
from src.data.splits import split_by_engine
from src.data.windows import create_windows
from src.models.conditioned_autoencoder import (
    RegimeConditionedAutoencoder,
)
from src.training.train_autoencoder import train_autoencoder
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run regime-conditioned autoencoder experiment."
    )

    parser.add_argument("--fd-id", default="FD001")
    parser.add_argument(
        "--experiment-name",
        default="fd001_ae_regime_conditioned_v001",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--n-regimes", type=int, default=6)
    parser.add_argument("--regime-embedding-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--healthy-frac", type=float, default=0.85)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--no-test-eval", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    train_df = load_train(args.fd_id)
    train_split, val_split = split_by_engine(
    train_df,
    val_frac=args.val_frac,
    seed=args.seed,
)

    train_healthy = select_healthy_region(
        train_split,
        healthy_frac=args.healthy_frac,
    )

    val_healthy = select_healthy_region(
        val_split,
        healthy_frac=args.healthy_frac,
    )

    condition_cols = list(SETTING_COLUMNS)

    feature_cols = [
        *SETTING_COLUMNS,
        *SENSOR_COLUMNS,
    ]

    regime_model = fit_regime_model(
        train_healthy,
        condition_cols=condition_cols,
        n_regimes=args.n_regimes,
        seed=args.seed,
    )

    train_healthy = train_healthy.copy()
    val_healthy = val_healthy.copy()

    train_healthy["operating_regime"] = assign_regimes(
        train_healthy,
        regime_model,
    )

    val_healthy["operating_regime"] = assign_regimes(
        val_healthy,
        regime_model,
    )

    normalization_stats = fit_regime_normalizer(
        train_healthy,
        feature_cols=feature_cols,
        regime_col="operating_regime",
    )

    train_healthy = train_healthy.copy()
    val_healthy = val_healthy.copy()

    train_healthy[feature_cols] = train_healthy[feature_cols].astype(float)
    val_healthy[feature_cols] = val_healthy[feature_cols].astype(float)

    train_normalized = transform_by_regime(
        train_healthy,
        normalization_stats,
    )

    val_normalized = transform_by_regime(
        val_healthy,
        normalization_stats,
    )

    train_windows = create_windows(
        train_normalized,
        window_size=args.window_size,
        stride=args.stride,
        feature_cols=feature_cols,
    )

    val_windows = create_windows(
        val_normalized,
        window_size=args.window_size,
        stride=args.stride,
        feature_cols=feature_cols,
    )

    train_X = torch.tensor(
        train_windows.X,
        dtype=torch.float32,
    )

    val_X = torch.tensor(
        val_windows.X,
        dtype=torch.float32,
    )

    train_settings = torch.tensor(
        train_windows.X[:, -1, :3],
        dtype=torch.float32,
    )

    val_settings = torch.tensor(
        val_windows.X[:, -1, :3],
        dtype=torch.float32,
    )

    model = RegimeConditionedAutoencoder(
        window_size=args.window_size,
        n_features=len(feature_cols),
        latent_dim=args.latent_dim,
        hidden_dims=(64, 32),
        regime_hidden_dims=(16,),
        regime_embedding_dim=args.regime_embedding_dim,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    loss_fn = torch.nn.MSELoss()

    history = {
        "train_losses": [],
        "val_losses": [],
        "best_epoch": -1,
        "stopped_early": False,
    }

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    patience = 10

    for epoch in range(args.epochs):
        model.train()

        permutation = torch.randperm(train_X.shape[0])

        train_loss_sum = 0.0
        train_n = 0

        for start in range(0, train_X.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]

            batch_x = train_X[indices]
            batch_settings = train_settings[indices]

            optimizer.zero_grad()

            reconstruction, _ = model(
                batch_x,
                batch_settings,
            )

            loss = loss_fn(
                reconstruction,
                batch_x,
            )

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.shape[0]
            train_n += batch_x.shape[0]

        train_loss = train_loss_sum / train_n

        model.eval()

        with torch.no_grad():
            val_reconstruction, _ = model(
                val_X,
                val_settings,
            )

            val_loss = loss_fn(
                val_reconstruction,
                val_X,
            ).item()

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }

            history["best_epoch"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            history["stopped_early"] = True
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    checkpoint_dir = Path("results") / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir / f"{args.experiment_name}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fd_id": args.fd_id,
            "window_size": args.window_size,
            "n_features": len(feature_cols),
            "latent_dim": args.latent_dim,
            "n_regimes": args.n_regimes,
            "regime_embedding_dim": args.regime_embedding_dim,
            "feature_cols": feature_cols,
            "regime_model": regime_model,
            "normalization_stats": normalization_stats,
        },
        checkpoint_path,
    )

    result = {
        "experiment_id": args.experiment_name,
        "dataset": args.fd_id,
        "window": {
            "size": args.window_size,
            "stride": args.stride,
        },
        "model": {
            "type": "regime_conditioned_autoencoder",
            "latent_dim": args.latent_dim,
            "hidden_dims": [64, 32],
            "regime_hidden_dims": [16],
            "regime_embedding_dim": args.regime_embedding_dim,
            "n_regimes": args.n_regimes,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "early_stopping_patience": patience,
        },
        "normalization": {
            "method": "per_regime_zscore",
        },
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "final_train_loss": history["train_losses"][-1],
        "final_val_loss": history["val_losses"][-1],
        "best_epoch": history["best_epoch"],
        "stopped_early": history["stopped_early"],
        "checkpoint_path": str(checkpoint_path),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()