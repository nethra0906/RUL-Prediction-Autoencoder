"""Experiment entry point: fixed-threshold baseline autoencoder.

See AI_CONTEXT.md Section 8.1 (baseline system) and Section 19
(`experiments/run_baseline.py` module contract). Ties together:

    load_train -> split_by_engine -> select_healthy_region
        -> fit_normalizer/transform -> create_windows -> detrend_windows
        -> train_autoencoder -> window_scores_numpy
        -> fit_fixed_threshold (absolute) + compute_expanding_delta_scores
        -> fit_fixed_threshold (delta)
        -> [TEST] load_test/load_test_rul -> label_anomalous_by_life_fraction
        -> transform (reusing TRAIN-fitted stats) -> create_windows -> detrend
        -> evaluate_detection, for BOTH absolute and delta scoring

DIAGNOSTIC FINDING (30 Aug 2026): the original (non-detrended) baseline
scored ROC-AUC ~0.29 on FD001 test — WORSE than random. Investigation
showed mean reconstruction error trending smoothly DOWNWARD across the
entire observed life range (not just at the anomaly boundary),
consistent with the dense AE finding smooth monotonic degradation
trends EASIER to reconstruct than stationary healthy noise. Two fixes
applied together in response:

  1. Per-window linear detrending (`data/detrend.py`) — removes each
     window's own linear trend before the model ever sees it, so a
     straight-line drift can no longer be "reconstructed for free".
  2. Per-engine expanding delta scoring (`anomaly/delta_score.py`) —
     an ADDITIONAL score reframing each window's error relative to
     that engine's own prior history, rather than as an absolute
     value, to also capture a rising trend in error itself.

Both are reported side-by-side (`test_evaluation_absolute` /
`test_evaluation_delta`) rather than one replacing the other, so the
actual effect of each is directly comparable on real results.

NOTE (see src/data/normalization.py): normalization here is a
temporary global z-score stub standing in for Akriti's planned
per-condition normalizer.

NOTE (see src/evaluation/evaluation_runner.py): `persistence` for
lead-time / event-level alerting defaults to 1 (first crossing) since
no team-wide value has been decided yet (AI_CONTEXT.md Section 12).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.anomaly.delta_score import compute_expanding_delta_scores
from src.anomaly.fixed_threshold import apply_fixed_threshold, fit_fixed_threshold
from src.anomaly.reconstruction import window_scores_numpy
from src.data.detrend import detrend_windows
from src.data.healthy_region import select_healthy_region
from src.data.loaders import load_test, load_test_rul, load_train
from src.data.normalization import fit_normalizer, transform
from src.data.schema import SENSOR_COLUMNS, SETTING_COLUMNS
from src.data.splits import split_by_engine
from src.data.windows import create_windows
from src.evaluation.evaluation_runner import (
    DetectionEvaluationResult,
    compute_engine_total_life,
    evaluate_detection,
    label_anomalous_by_life_fraction,
)
from src.models.autoencoder import DenseAutoencoder
from src.training.train_autoencoder import train_autoencoder

_FEATURE_COLS = [*SETTING_COLUMNS, *SENSOR_COLUMNS]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _eval_result_to_dict(eval_result: DetectionEvaluationResult, n_windows: int) -> dict:
    return {
        "n_test_windows": n_windows,
        "precision": eval_result.precision,
        "recall": eval_result.recall,
        "f1": eval_result.f1,
        "roc_auc": eval_result.roc_auc,
        "false_alarm_rate": eval_result.false_alarm_rate,
        "detection_rate": eval_result.detection_rate,
        "mean_lead_time": eval_result.mean_lead_time,
        "n_engines": eval_result.n_engines,
    }


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
    delta_threshold_lambda: float = 2.5,
    delta_baseline_window: int | None = None,
    persistence: int = 1,
    detrend: bool = True,
    seed: int = 42,
    experiment_name: str = "fd001_ae_fixed_v002_detrend_delta",
    evaluate_on_test: bool = True,
) -> dict:
    """Run the full baseline pipeline end-to-end on one C-MAPSS subset.

    Args:
        detrend: If True (default), apply `data.detrend.detrend_windows`
            to every window (train/val/test) before it reaches the
            model. See module docstring for why this was introduced.
        delta_threshold_lambda: Separate lambda for the delta-score
            threshold (delta scores are typically much smaller in
            magnitude than absolute reconstruction error, so a
            different multiplier may be appropriate — tune independently).
        delta_baseline_window: Passed to
            `anomaly.delta_score.compute_expanding_delta_scores`. None
            (default) uses an expanding mean from each engine's
            observed trajectory start; an int uses a rolling window of
            that many prior windows instead.
        evaluate_on_test: If True (default), also loads test data,
            labels anomalous windows via `healthy_frac`-based life
            fraction, and computes the full detection metric suite for
            BOTH absolute and delta scoring.

    Returns:
        A dict summary (also written to `results/logs/<experiment_name>.json`).
    """
    config = {
        "experiment_id": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": fd_id,
        "split": {"val_frac": val_frac, "seed": seed},
        "healthy_region": {"healthy_frac": healthy_frac},
        "window": {"size": window_size, "stride": stride},
        "detrend": detrend,
        "model": {"type": "baseline_autoencoder", "latent_dim": latent_dim, "hidden_dims": list(hidden_dims)},
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "early_stopping_patience": early_stopping_patience,
        },
        "threshold": {
            "method": "fixed",
            "lambda_absolute": threshold_lambda,
            "lambda_delta": delta_threshold_lambda,
            "delta_baseline_window": delta_baseline_window,
        },
        "evaluation": {"persistence": persistence},
    }

    # 1. Load + engine-level split (AI_CONTEXT.md Section 17 Rule 1).
    train_full = load_train(fd_id)
    train_split, val_split = split_by_engine(train_full, val_frac=val_frac, seed=seed)

    # 2. Healthy-region selection, per split, independently.
    train_healthy = select_healthy_region(train_split, healthy_frac=healthy_frac)
    val_healthy = select_healthy_region(val_split, healthy_frac=healthy_frac)

    # 3. Normalization: fit on train_healthy ONLY (AI_CONTEXT.md Section 17 Rule 2).
    norm_stats = fit_normalizer(train_healthy, feature_cols=_FEATURE_COLS)
    train_norm = transform(train_healthy, norm_stats)
    val_norm = transform(val_healthy, norm_stats)

    # 4. Windowing (engine-boundary safe by construction).
    train_windows = create_windows(train_norm, window_size=window_size, stride=stride, feature_cols=_FEATURE_COLS)
    val_windows = create_windows(val_norm, window_size=window_size, stride=stride, feature_cols=_FEATURE_COLS)

    if len(train_windows) == 0 or len(val_windows) == 0:
        raise ValueError(
            f"window_size={window_size} produced zero windows for one or both "
            "splits — reduce window_size or check healthy_frac isn't cutting "
            "trajectories too short."
        )

    train_X_np = train_windows.X
    val_X_np = val_windows.X
    if detrend:
        train_X_np = detrend_windows(train_X_np)
        val_X_np = detrend_windows(val_X_np)

    train_X = torch.tensor(train_X_np, dtype=torch.float32)
    val_X = torch.tensor(val_X_np, dtype=torch.float32)

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

    # 6. Absolute reconstruction error + fixed threshold, fit on healthy
    #    TRAIN scores only (AI_CONTEXT.md Section 17 Rule 3).
    model.eval()
    with torch.no_grad():
        train_recon, _ = model(train_X)
    train_scores = window_scores_numpy(train_X, train_recon)
    abs_threshold = fit_fixed_threshold(train_scores, lambda_=threshold_lambda)

    with torch.no_grad():
        val_recon, _ = model(val_X)
    val_scores = window_scores_numpy(val_X, val_recon)
    val_alerts_abs = apply_fixed_threshold(val_scores, abs_threshold)

    # 6b. Delta scoring: reframe train/val absolute scores relative to
    #     each engine's own prior history, then fit a SEPARATE threshold
    #     on healthy TRAIN delta scores (still never touches test data).
    train_delta_scores = compute_expanding_delta_scores(
        train_scores, train_windows.engine_ids, train_windows.end_cycles, delta_baseline_window
    )
    delta_threshold = fit_fixed_threshold(train_delta_scores, lambda_=delta_threshold_lambda)

    val_delta_scores = compute_expanding_delta_scores(
        val_scores, val_windows.engine_ids, val_windows.end_cycles, delta_baseline_window
    )
    val_alerts_delta = apply_fixed_threshold(val_delta_scores, delta_threshold)

    summary = {
        **config,
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "final_train_loss": history.train_losses[-1],
        "final_val_loss": history.val_losses[-1],
        "best_epoch": history.best_epoch,
        "stopped_early": history.stopped_early,
        "threshold_value_absolute": abs_threshold.value,
        "threshold_value_delta": delta_threshold.value,
        "val_alert_rate_absolute": float(val_alerts_abs.mean()),
        "val_alert_rate_delta": float(val_alerts_delta.mean()),
    }

    # 7. TEST-SET evaluation (AI_CONTEXT.md Section 17 Rule 5: test RUL
    #    used only to build evaluation labels, never touches training
    #    or threshold fitting above).
    if evaluate_on_test:
        test_df = load_test(fd_id)
        test_rul = load_test_rul(fd_id)

        test_labeled = label_anomalous_by_life_fraction(test_df, test_rul, healthy_frac=healthy_frac)
        test_norm = transform(test_labeled, norm_stats)  # reuse TRAIN-fitted stats
        test_windows = create_windows(
            test_norm,
            window_size=window_size,
            stride=stride,
            feature_cols=_FEATURE_COLS,
            label_cols=["is_anomalous"],
        )

        if len(test_windows) == 0:
            summary["test_evaluation_warning"] = (
                f"window_size={window_size} produced zero test windows."
            )
        else:
            test_X_np = test_windows.X
            if detrend:
                test_X_np = detrend_windows(test_X_np)
            test_X = torch.tensor(test_X_np, dtype=torch.float32)

            with torch.no_grad():
                test_recon, _ = model(test_X)
            test_scores = window_scores_numpy(test_X, test_recon)
            test_alerts_abs = apply_fixed_threshold(test_scores, abs_threshold)
            test_y_true = test_windows.y.flatten()

            test_delta_scores = compute_expanding_delta_scores(
                test_scores, test_windows.engine_ids, test_windows.end_cycles, delta_baseline_window
            )
            test_alerts_delta = apply_fixed_threshold(test_delta_scores, delta_threshold)

            total_life = compute_engine_total_life(test_df, test_rul)

            eval_absolute = evaluate_detection(
                y_true=test_y_true,
                scores=test_scores,
                alerts=test_alerts_abs,
                engine_ids=test_windows.engine_ids,
                end_cycles=test_windows.end_cycles,
                engine_total_life=total_life,
                persistence=persistence,
            )
            eval_delta = evaluate_detection(
                y_true=test_y_true,
                scores=test_delta_scores,
                alerts=test_alerts_delta,
                engine_ids=test_windows.engine_ids,
                end_cycles=test_windows.end_cycles,
                engine_total_life=total_life,
                persistence=persistence,
            )

            summary["test_evaluation_absolute"] = _eval_result_to_dict(eval_absolute, len(test_windows))
            summary["test_evaluation_delta"] = _eval_result_to_dict(eval_delta, len(test_windows))

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
    parser.add_argument("--experiment-name", default="fd001_ae_fixed_v002_detrend_delta")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-detrend", action="store_true", help="Disable per-window detrending.")
    parser.add_argument("--no-test-eval", action="store_true", help="Skip test-set evaluation.")
    args = parser.parse_args()

    summary = run_baseline_experiment(
        fd_id=args.fd_id,
        epochs=args.epochs,
        seed=args.seed,
        experiment_name=args.experiment_name,
        detrend=not args.no_detrend,
        evaluate_on_test=not args.no_test_eval,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
