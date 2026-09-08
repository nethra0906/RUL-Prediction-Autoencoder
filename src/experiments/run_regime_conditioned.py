from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.anomaly.fixed_threshold import apply_fixed_threshold, fit_fixed_threshold
from src.anomaly.reconstruction import normalized_anomaly_scores, window_scores_numpy
from src.data.healthy_region import select_healthy_region
from src.data.loaders import load_test, load_test_rul, load_train
from src.data.schema import SENSOR_COLUMNS, SETTING_COLUMNS
from src.data.normalization import fit_regime_normalizer, transform_by_regime
from src.data.regimes import assign_regimes, fit_regime_model
from src.data.splits import split_by_engine
from src.data.windows import create_windows
from src.evaluation.evaluation_runner import (
    compute_engine_total_life,
    evaluate_detection,
    label_anomalous_by_life_fraction,
)
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
    parser.add_argument("--threshold-lambda", type=float, default=2.5)
    parser.add_argument(
        "--target-alert-rate",
        type=float,
        default=0.03,
        help="Used to auto-select lambda from a sweep via val alert rate.",
    )
    parser.add_argument("--persistence", type=int, default=1)
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

    # ------------------------------------------------------------------
    # Evaluation: reconstruction scoring + fixed-threshold lambda sweep
    # + full test-set evaluation.
    #
    # NOTE: uses the FIXED threshold approach (proven working on the
    # baseline experiment), not the rolling dynamic threshold (found to
    # adapt away the very degradation signal it should catch -- see
    # src/anomaly/dynamic_threshold.py module docstring, 3 Sep 2026).
    #
    # NOTE: scoring uses `normalized_anomaly_scores`, since raw
    # reconstruction error was found to invert relative to the true
    # anomaly label on the vanilla DenseAutoencoder (see
    # src/anomaly/reconstruction.py module docstring). This model
    # reuses the same DenseAutoencoder encoder/decoder internally, so
    # the same correction is applied here for consistency; if this
    # assumption turns out not to hold for the conditioned model, the
    # diagnostics in scripts/diagnose_test_scores.py should be re-run
    # against this checkpoint specifically before trusting these
    # numbers.
    # ------------------------------------------------------------------

    model.eval()

    with torch.no_grad():
        train_recon, _ = model(train_X, train_settings)
    train_raw_scores = window_scores_numpy(train_X, train_recon)
    train_scores = normalized_anomaly_scores(train_windows.X, train_raw_scores)

    with torch.no_grad():
        val_recon, _ = model(val_X, val_settings)
    val_raw_scores = window_scores_numpy(val_X, val_recon)
    val_scores = normalized_anomaly_scores(val_windows.X, val_raw_scores)

    # Lambda sweep, selected via TRAIN (fit) + VAL (alert-rate check)
    # only -- never against test performance (AI_CONTEXT.md Section 17
    # Rule 3). Mirrors the sweep used in the corrected baseline
    # experiment, since a fixed lambda=2.5 was previously found to
    # produce val_alert_rate=0.0 after the scoring correction changed
    # the score distribution's scale/shape.
    candidate_lambdas = [0.5, 1.0, 1.5, 2.0, 2.5]
    sweep_results = []
    best_lambda, best_threshold, best_gap = None, None, None
    for lam in candidate_lambdas:
        cand_threshold = fit_fixed_threshold(train_scores, lambda_=lam)
        cand_val_alerts = apply_fixed_threshold(val_scores, cand_threshold)
        alert_rate = float(cand_val_alerts.mean())
        gap = abs(alert_rate - args.target_alert_rate)
        sweep_results.append(
            {"lambda": lam, "threshold": cand_threshold.value, "val_alert_rate": alert_rate}
        )
        print(f"lambda={lam}: threshold={cand_threshold.value:.4f}, val_alert_rate={alert_rate:.4f}")
        if best_gap is None or gap < best_gap:
            best_gap, best_lambda, best_threshold = gap, lam, cand_threshold

    threshold = best_threshold
    val_alerts = apply_fixed_threshold(val_scores, threshold)
    print(
        f"\nSelected lambda={best_lambda} (threshold={threshold.value:.4f}) "
        f"-- closest val_alert_rate to target {args.target_alert_rate}"
    )

    test_evaluation = None
    if not args.no_test_eval:
        test_df = load_test(args.fd_id)
        test_rul = load_test_rul(args.fd_id)

        test_labeled = label_anomalous_by_life_fraction(
            test_df,
            test_rul,
            healthy_frac=args.healthy_frac,
        )

        test_labeled = test_labeled.copy()
        test_labeled["operating_regime"] = assign_regimes(test_labeled, regime_model)
        test_labeled[feature_cols] = test_labeled[feature_cols].astype(float)
        test_normalized = transform_by_regime(test_labeled, normalization_stats)

        test_windows = create_windows(
            test_normalized,
            window_size=args.window_size,
            stride=args.stride,
            feature_cols=feature_cols,
            label_cols=["is_anomalous"],
        )

        if len(test_windows) == 0:
            test_evaluation = {
                "warning": f"window_size={args.window_size} produced zero test windows."
            }
        else:
            test_X = torch.tensor(test_windows.X, dtype=torch.float32)
            test_settings = torch.tensor(test_windows.X[:, -1, :3], dtype=torch.float32)

            with torch.no_grad():
                test_recon, _ = model(test_X, test_settings)
            test_raw_scores = window_scores_numpy(test_X, test_recon)
            test_scores = normalized_anomaly_scores(test_windows.X, test_raw_scores)
            test_alerts = apply_fixed_threshold(test_scores, threshold)
            test_y_true = test_windows.y.flatten()

            print(f"test_alerts.sum() = {test_alerts.sum()} / {len(test_alerts)}")
            print(f"test_y_true.sum() = {test_y_true.sum()} / {len(test_y_true)}")
            print(f"overlap (alert AND true) = {(test_alerts & test_y_true.astype(bool)).sum()}")

            total_life = compute_engine_total_life(test_df, test_rul)
            eval_result = evaluate_detection(
                y_true=test_y_true,
                scores=test_scores,
                alerts=test_alerts,
                engine_ids=test_windows.engine_ids,
                end_cycles=test_windows.end_cycles,
                engine_total_life=total_life,
                persistence=args.persistence,
            )

            test_evaluation = {
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

    # ------------------------------------------------------------------
    # Save checkpoint and experiment log.
    # ------------------------------------------------------------------

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
            "threshold_value": threshold.value,
        },
        checkpoint_path,
    )

    logs_dir = Path("results") / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{args.experiment_name}.json"

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
        "threshold": {
            "method": "fixed",
            "selected_lambda": best_lambda,
            "value": threshold.value,
            "sweep": sweep_results,
        },
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
        "final_train_loss": history["train_losses"][-1],
        "final_val_loss": history["val_losses"][-1],
        "best_epoch": history["best_epoch"],
        "stopped_early": history["stopped_early"],
        "val_alert_rate": float(val_alerts.mean()),
        "test_evaluation": test_evaluation,
        "checkpoint_path": str(checkpoint_path),
    }

    log_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()