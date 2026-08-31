"""Experiment entry point: fixed-threshold baseline autoencoder.

See AI_CONTEXT.md Section 8.1 (baseline system) and Section 19
(`experiments/run_baseline.py` module contract). Ties together:

    load_train -> split_by_engine -> select_healthy_region
        -> fit_normalizer/transform -> create_windows
        -> train_autoencoder -> window_scores_numpy
        -> fit_fixed_threshold
        -> [TEST] load_test/load_test_rul -> label_anomalous_by_life_fraction
        -> transform (reusing TRAIN-fitted stats) -> create_windows
        -> evaluate_detection (Precision/Recall/F1/ROC-AUC/lead time)

Per AI_CONTEXT.md Section 21, every run should log its configuration,
split, seed, and metrics. This script writes a JSON summary to
`results/logs/` and the trained weights to `results/checkpoints/`,
named with the experiment name convention from Section 22.

Test-set anomaly labeling convention (decided 30 Aug 2026, see
`evaluation/evaluation_runner.py` module docstring): a test cycle is
anomalous if it falls beyond `healthy_frac` of that engine's TRUE total
life (observed cycles + ground-truth RUL) — the same `healthy_frac`
used to select the training healthy region, so both boundaries are
defined consistently. The test RUL file is used ONLY to build this
evaluation label, never as model input or for threshold selection
(AI_CONTEXT.md Section 17 Rule 5).

NOTE (see src/data/normalization.py): normalization here is a
temporary global z-score stub standing in for Akriti's planned
per-condition normalizer. Swapping her version in only requires
changing the `fit_normalizer`/`transform` import in this file. The
same fitted stats (from healthy TRAIN data only) are reused to
transform the test set, per AI_CONTEXT.md Section 17 Rule 2.

NOTE (see src/evaluation/evaluation_runner.py): `persistence` for the
lead-time / event-level alert rule defaults to 1 (first threshold
crossing counts as an alert) since no team-wide value has been decided
yet (AI_CONTEXT.md Section 12) — override once one is.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.anomaly.fixed_threshold import apply_fixed_threshold, fit_fixed_threshold
from src.anomaly.reconstruction import window_scores_numpy
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


def run_baseline_experiment(
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
    threshold_lambda: float = 2.5,
    persistence: int = 1,
    seed: int = 42,
    experiment_name: str = "fd001_ae_fixed_v001",
    evaluate_on_test: bool = True,
) -> dict:
    """Run the full baseline pipeline end-to-end on one C-MAPSS subset.

    All arguments default to `configs/base.yaml` / `configs/baseline_ae.yaml`
    values as of this checkpoint, but should be passed explicitly by
    callers reading from an actual config file rather than relying on
    these defaults for a logged experiment (AI_CONTEXT.md Section 20).

    Args:
        evaluate_on_test: If True (default), also loads `test_{fd_id}.txt`
            + `RUL_{fd_id}.txt`, labels anomalous windows via
            `healthy_frac`-based life-fraction, and computes the full
            detection metric suite. Set False to reproduce the
            train/val-only behavior of the original baseline run.

    Returns:
        A dict summary (also written to `results/logs/<experiment_name>.json`)
        containing the config used, dataset split sizes, final train/val
        loss, the fitted threshold, checkpoint path, and — if
        `evaluate_on_test` — full test-set detection metrics. This is
        the minimum AI_CONTEXT.md Section 21 requires for a run to
        count as reproducible.
    """
    config = {
        "experiment_id": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": fd_id,
        "split": {"val_frac": val_frac, "seed": seed},
        "healthy_region": {"healthy_frac": healthy_frac},
        "window": {"size": window_size, "stride": stride},
        "model": {"type": "baseline_autoencoder", "latent_dim": latent_dim, "hidden_dims": list(hidden_dims)},
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "early_stopping_patience": early_stopping_patience,
        },
        "threshold": {"method": "fixed", "lambda": threshold_lambda},
        "evaluation": {"persistence": persistence},
    }

    # 1. Load + engine-level split (AI_CONTEXT.md Section 17 Rule 1).
    train_full = load_train(fd_id)
    train_split, val_split = split_by_engine(train_full, val_frac=val_frac, seed=seed)

    # 2. Healthy-region selection, per split, independently.
    train_healthy = select_healthy_region(train_split, healthy_frac=healthy_frac)
    val_healthy = select_healthy_region(val_split, healthy_frac=healthy_frac)

    # 3. Normalization: fit on train_healthy ONLY, apply to both
    #    (AI_CONTEXT.md Section 17 Rule 2).
    norm_stats = fit_normalizer(train_healthy, feature_cols=_FEATURE_COLS)
    train_norm = transform(train_healthy, norm_stats)
    val_norm = transform(val_healthy, norm_stats)

    # 4. Windowing (engine-boundary safe by construction).
    train_windows = create_windows(train_norm, window_size=window_size, stride=stride, feature_cols=_FEATURE_COLS)
    val_windows = create_windows(val_norm, window_size=window_size, stride=stride, feature_cols=_FEATURE_COLS)

    if len(train_windows) == 0 or len(val_windows) == 0:
        raise ValueError(
            f"window_size={window_size} produced zero windows for one or both "
            "splits — reduce window_size or check the healthy_frac isn't "
            "cutting trajectories too short."
        )

    train_X = torch.tensor(train_windows.X, dtype=torch.float32)
    val_X = torch.tensor(val_windows.X, dtype=torch.float32)

    # 5. Build + train the model.
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

    # 6. Reconstruction error + fixed threshold, fit on healthy TRAIN
    #    scores only (AI_CONTEXT.md Section 17 Rule 3 — test scores
    #    never touch threshold fitting).
    model.eval()
    with torch.no_grad():
        train_recon, _ = model(train_X)
    train_scores = window_scores_numpy(train_X, train_recon)
    threshold = fit_fixed_threshold(train_scores, lambda_=threshold_lambda)

    with torch.no_grad():
        val_recon, _ = model(val_X)
    val_scores = window_scores_numpy(val_X, val_recon)
    val_alerts = apply_fixed_threshold(val_scores, threshold)

    summary = {
        **config,
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "final_train_loss": history.train_losses[-1],
        "final_val_loss": history.val_losses[-1],
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "threshold_value": threshold.value,
        "threshold_mean": threshold.mean,
        "threshold_std": threshold.std,
        "val_alert_rate": float(val_alerts.mean()),  # false-alarm rate on healthy val data
    }

    # 7. TEST-SET evaluation (AI_CONTEXT.md Section 17 Rule 5: test RUL
    #    used only to build evaluation labels here, never touches
    #    training or threshold fitting above).
    if evaluate_on_test:
        test_df = load_test(fd_id)
        test_rul = load_test_rul(fd_id)

        test_labeled = label_anomalous_by_life_fraction(
            test_df, test_rul, healthy_frac=healthy_frac
        )
        test_norm = transform(test_labeled, norm_stats)  # reuse TRAIN-fitted stats
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
                    f"window_size={window_size} produced zero test windows "
                    "(some test trajectories may be shorter than window_size)."
                )
            }
        else:
            test_X = torch.tensor(test_windows.X, dtype=torch.float32)
            with torch.no_grad():
                test_recon, _ = model(test_X)
            test_scores = window_scores_numpy(test_X, test_recon)
            test_alerts = apply_fixed_threshold(test_scores, threshold)
            test_y_true = test_windows.y.flatten()

            total_life = compute_engine_total_life(test_df, test_rul)

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
            }

    # 8. Persist checkpoint + config/results log (AI_CONTEXT.md Section 21).
    checkpoints_dir = _REPO_ROOT / "results" / "checkpoints"
    logs_dir = _REPO_ROOT / "results" / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoints_dir / f"{experiment_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    summary["checkpoint_path"] = str(checkpoint_path.relative_to(_REPO_ROOT))

    log_path = logs_dir / f"{experiment_name}.json"
    log_path.write_text(json.dumps(summary, indent=2))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd-id", default="FD001")
    parser.add_argument("--experiment-name", default="fd001_ae_fixed_v001")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-test-eval", action="store_true", help="Skip test-set evaluation.")
    args = parser.parse_args()

    summary = run_baseline_experiment(
        fd_id=args.fd_id,
        epochs=args.epochs,
        seed=args.seed,
        experiment_name=args.experiment_name,
        evaluate_on_test=not args.no_test_eval,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
