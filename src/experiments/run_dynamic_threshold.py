from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.anomaly.dynamic_threshold import (
    apply_dynamic_threshold,
    fit_dynamic_threshold_per_engine,
)
from src.anomaly.reconstruction import normalized_anomaly_scores, window_scores_numpy
from src.data.healthy_region import select_healthy_region
from src.data.loaders import load_test, load_test_rul, load_train
from src.data.normalization import fit_normalizer, transform
from src.data.schema import SENSOR_COLUMNS, SETTING_COLUMNS
from src.data.splits import split_by_engine
from src.data.windows import create_windows
from src.evaluation.evaluation_runner import (
    compute_engine_total_life,
    evaluate_detection,
    label_anomalous_by_life_fraction,
)
from src.models.autoencoder import DenseAutoencoder
from src.training.train_autoencoder import train_autoencoder


_FEATURE_COLS = [*SETTING_COLUMNS, *SENSOR_COLUMNS]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def run_dynamic_threshold_experiment(
    fd_id: str = "FD001",
    val_frac: float = 0.15,
    healthy_frac: float = 0.85,
    window_size: int = 30,
    stride: int = 1,
    latent_dim: int = 16,
    hidden_dims: tuple[int, ...] = (64, 32),
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    early_stopping_patience: int | None = 10,
    threshold_window_size: int = 30,
    threshold_lambda: float = 2.5,
    persistence: int = 1,
    seed: int = 42,
    experiment_name: str = "fd001_ae_dynamic_v001",
    evaluate_on_test: bool = True,
) -> dict:
    """Run the dynamic-threshold autoencoder experiment."""

    config = {
        "experiment_id": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": fd_id,
        "split": {
            "val_frac": val_frac,
            "seed": seed,
        },
        "healthy_region": {
            "healthy_frac": healthy_frac,
        },
        "window": {
            "size": window_size,
            "stride": stride,
        },
        "model": {
            "type": "baseline_autoencoder",
            "latent_dim": latent_dim,
            "hidden_dims": list(hidden_dims),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "early_stopping_patience": early_stopping_patience,
        },
        "threshold": {
            "method": "dynamic",
            "window_size": threshold_window_size,
            "lambda": threshold_lambda,
        },
        "evaluation": {
            "persistence": persistence,
        },
    }

    # 1. Load data and split by engine.
    train_full = load_train(fd_id)

    train_split, val_split = split_by_engine(
        train_full,
        val_frac=val_frac,
        seed=seed,
    )

    # 2. Select healthy regions.
    train_healthy = select_healthy_region(
        train_split,
        healthy_frac=healthy_frac,
    )

    val_healthy = select_healthy_region(
        val_split,
        healthy_frac=healthy_frac,
    )

    # 3. Fit normalization only on healthy training data.
    norm_stats = fit_normalizer(
        train_healthy,
        feature_cols=_FEATURE_COLS,
    )

    train_norm = transform(
        train_healthy,
        norm_stats,
    )

    val_norm = transform(
        val_healthy,
        norm_stats,
    )

    # 4. Create windows.
    train_windows = create_windows(
        train_norm,
        window_size=window_size,
        stride=stride,
        feature_cols=_FEATURE_COLS,
    )

    val_windows = create_windows(
        val_norm,
        window_size=window_size,
        stride=stride,
        feature_cols=_FEATURE_COLS,
    )

    if len(train_windows) == 0 or len(val_windows) == 0:
        raise ValueError(
            f"window_size={window_size} produced zero windows for "
            "one or both splits."
        )

    train_X = torch.tensor(
        train_windows.X,
        dtype=torch.float32,
    )

    val_X = torch.tensor(
        val_windows.X,
        dtype=torch.float32,
    )

    # 5. Build and train the autoencoder.
    model = DenseAutoencoder(
        window_size=window_size,
        n_features=len(_FEATURE_COLS),
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )

    model, history = train_autoencoder(
        model,
        train_X=train_X,
        val_X=val_X,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        early_stopping_patience=early_stopping_patience,
    )

    model.eval()

    # 6. Compute reconstruction scores.
    with torch.no_grad():
        train_recon, _ = model(train_X)

    train_scores = window_scores_numpy(
        train_X,
        train_recon,
    )
    train_scores = normalized_anomaly_scores(train_windows.X, train_scores)

    with torch.no_grad():
        val_recon, _ = model(val_X)

    val_scores = window_scores_numpy(
        val_X,
        val_recon,
    )
    val_scores = normalized_anomaly_scores(val_windows.X, val_scores)

    # 7. Dynamic threshold on the healthy validation stream.
    val_thresholds = fit_dynamic_threshold_per_engine(
        val_scores,
        engine_ids=val_windows.engine_ids,
        end_cycles=val_windows.end_cycles,
        window_size=threshold_window_size,
        lambda_=threshold_lambda,
    )

    val_alerts = apply_dynamic_threshold(
        val_scores,
        val_thresholds,
    )

    summary = {
        **config,
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "final_train_loss": history.train_losses[-1],
        "final_val_loss": history.val_losses[-1],
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "val_alert_rate": float(val_alerts.mean()),
        "val_threshold_mean": float(val_thresholds.mean()),
        "val_threshold_min": float(val_thresholds.min()),
        "val_threshold_max": float(val_thresholds.max()),
    }

    # 8. Test-set evaluation.
    if evaluate_on_test:
        test_df = load_test(fd_id)
        test_rul = load_test_rul(fd_id)

        test_labeled = label_anomalous_by_life_fraction(
            test_df,
            test_rul,
            healthy_frac=healthy_frac,
        )

        test_norm = transform(
            test_labeled,
            norm_stats,
        )

        test_windows = create_windows(
            test_norm,
            window_size=window_size,
            stride=stride,
            feature_cols=_FEATURE_COLS,
            label_cols=["is_anomalous"],
        )
        

        if len(test_windows) == 0:
            summary["test_evaluation"] = {
                "warning": (
                    f"window_size={window_size} produced zero test windows."
                )
            }
        else:
            test_X = torch.tensor(
                test_windows.X,
                dtype=torch.float32,
            )

            with torch.no_grad():
                test_recon, _ = model(test_X)

            test_scores = window_scores_numpy(
                test_X,
                test_recon,
            )
            test_scores = normalized_anomaly_scores(test_windows.X, test_scores)

            # Dynamic threshold is generated sequentially from the
            # test score stream, without using test labels.
            test_thresholds = fit_dynamic_threshold_per_engine(
                test_scores,
                engine_ids=test_windows.engine_ids,
                end_cycles=test_windows.end_cycles,
                window_size=threshold_window_size,
                lambda_=threshold_lambda,
            )

            test_alerts = apply_dynamic_threshold(
                test_scores,
                test_thresholds,
            )

            test_y_true = test_windows.y.flatten()

            total_life = compute_engine_total_life(
                test_df,
                test_rul,
            )
            print(f"test_alerts.sum() = {test_alerts.sum()} / {len(test_alerts)}")
            print(f"test_y_true.sum() = {test_y_true.sum()} / {len(test_y_true)}")
            print(f"overlap (alert AND true) = {(test_alerts & test_y_true.astype(bool)).sum()}")
            print(f"alerts on healthy = {(test_alerts & ~test_y_true.astype(bool)).sum()}")

            eval_result = evaluate_detection(
                y_true=test_y_true,
                scores=test_scores,
                alerts=test_alerts,
                engine_ids=test_windows.engine_ids,
                end_cycles=test_windows.end_cycles,
                engine_total_life=total_life,
                persistence=persistence,
            )

            summary["test_evaluation"] = {
                "n_test_windows": len(test_windows),
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "f1": eval_result.f1,
                "roc_auc": eval_result.roc_auc,
                "false_alarm_rate": eval_result.false_alarm_rate,
                "detection_rate": eval_result.detection_rate,
                "mean_lead_time": eval_result.mean_lead_time,
                "n_engines": eval_result.n_engines,
                "threshold_mean": float(test_thresholds.mean()),
                "threshold_min": float(test_thresholds.min()),
                "threshold_max": float(test_thresholds.max()),
            }

    # 9. Save checkpoint and experiment log.
    checkpoints_dir = _REPO_ROOT / "results" / "checkpoints"
    logs_dir = _REPO_ROOT / "results" / "logs"

    checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoints_dir / f"{experiment_name}.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    summary["checkpoint_path"] = str(
        checkpoint_path.relative_to(_REPO_ROOT)
    )

    log_path = logs_dir / f"{experiment_name}.json"

    log_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--fd-id",
        default="FD001",
    )

    parser.add_argument(
        "--experiment-name",
        default="fd001_ae_dynamic_v001",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--threshold-window-size",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--threshold-lambda",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--no-test-eval",
        action="store_true",
        help="Skip test-set evaluation.",
    )

    args = parser.parse_args()

    summary = run_dynamic_threshold_experiment(
        fd_id=args.fd_id,
        epochs=args.epochs,
        seed=args.seed,
        experiment_name=args.experiment_name,
        threshold_window_size=args.threshold_window_size,
        threshold_lambda=args.threshold_lambda,
        evaluate_on_test=not args.no_test_eval,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()