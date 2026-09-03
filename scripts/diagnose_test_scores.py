"""One-off diagnostic: inspect test-set score/label separation for the
baseline run, to debug why ROC-AUC is below 0.5 (see run_baseline.py).

UPDATED (31 Aug 2026): now scores BOTH raw and detrended test windows
through the (detrend-trained) checkpoint, and reports per-window input
variance alongside score, to distinguish two competing hypotheses:

  H1: detrending destroyed the genuine anomaly signal
      -> raw scores should separate healthy/anomalous better than
         detrended scores.
  H2: the autoencoder is a degenerate/near-mean-predictor and score is
      just tracking input variance, not learned structure
      -> score_detrended should correlate strongly with input variance
         regardless of true label; raw-vs-detrended should look similar
         in shape.

Not part of the permanent pipeline -- a throwaway investigation script.
Run from repo root: `python scripts/diagnose_test_scores.py`
"""

import numpy as np
import torch
from scipy.stats import spearmanr

from src.anomaly.reconstruction import window_scores_numpy
from src.data.detrend import detrend_windows
from src.data.healthy_region import select_healthy_region
from src.data.loaders import load_test, load_test_rul, load_train
from src.data.normalization import fit_normalizer, transform
from src.data.schema import SENSOR_COLUMNS, SETTING_COLUMNS
from src.data.splits import split_by_engine
from src.data.windows import create_windows
from src.evaluation.evaluation_runner import (
    compute_engine_total_life,
    label_anomalous_by_life_fraction,
)
from src.models.autoencoder import DenseAutoencoder

FEATURE_COLS = [*SETTING_COLUMNS, *SENSOR_COLUMNS]
FD_ID = "FD001"
WINDOW_SIZE = 30
STRIDE = 1
HEALTHY_FRAC = 0.85
# NOTE: point this at whichever checkpoint you want to diagnose. The
# current baseline (run_baseline.py) trains on DETRENDED windows, so
# "raw" scores below reflect a detrend-trained model fed non-detrended
# input -- still informative, but read the printed labels carefully.
CHECKPOINT_PATH = "results/checkpoints/fd001_ae_fixed_v003_nodetrend.pt"

# Rebuild the same train-fitted normalization stats used by run_baseline.
train_full = load_train(FD_ID)
train_split, _ = split_by_engine(train_full, val_frac=0.15, seed=42)
train_healthy = select_healthy_region(train_split, healthy_frac=HEALTHY_FRAC)
norm_stats = fit_normalizer(train_healthy, feature_cols=FEATURE_COLS)

# Reload the trained model checkpoint.
model = DenseAutoencoder(window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS), latent_dim=16, hidden_dims=(64, 32))
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()
print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

# Rebuild test windows + labels exactly as run_baseline does.
test_df = load_test(FD_ID)
test_rul = load_test_rul(FD_ID)
test_labeled = label_anomalous_by_life_fraction(test_df, test_rul, healthy_frac=HEALTHY_FRAC)
test_norm = transform(test_labeled, norm_stats)
test_windows = create_windows(
    test_norm, window_size=WINDOW_SIZE, stride=STRIDE, feature_cols=FEATURE_COLS, label_cols=["is_anomalous"]
)

y_true = test_windows.y.flatten().astype(bool)

# --- Score BOTH raw and detrended variants through the same model ---
test_X_raw = torch.tensor(test_windows.X, dtype=torch.float32)
test_X_detrended = torch.tensor(detrend_windows(test_windows.X), dtype=torch.float32)

with torch.no_grad():
    recon_raw, _ = model(test_X_raw)
    recon_detrended, _ = model(test_X_detrended)

scores_raw = window_scores_numpy(test_X_raw, recon_raw)
scores_detrended = window_scores_numpy(test_X_detrended, recon_detrended)

# Per-window input variance, computed on what the model actually saw
# for the detrended pass (over time and channel axes).
input_variance_detrended = test_X_detrended.numpy().var(axis=(1, 2))
input_variance_raw = test_X_raw.numpy().var(axis=(1, 2))

print(f"\nTotal test windows: {len(y_true)}")
print(f"Anomalous windows: {y_true.sum()} ({y_true.mean():.2%})")

print("\n--- Score stats: RAW (non-detrended) input ---")
print(f"  healthy:   mean={scores_raw[~y_true].mean():.4f}  std={scores_raw[~y_true].std():.4f}")
if y_true.sum() > 0:
    print(f"  anomalous: mean={scores_raw[y_true].mean():.4f}  std={scores_raw[y_true].std():.4f}")

print("\n--- Score stats: DETRENDED input ---")
print(f"  healthy:   mean={scores_detrended[~y_true].mean():.4f}  std={scores_detrended[~y_true].std():.4f}")
if y_true.sum() > 0:
    print(f"  anomalous: mean={scores_detrended[y_true].mean():.4f}  std={scores_detrended[y_true].std():.4f}")

# How many engines actually reach the anomalous region within their
# OBSERVED (truncated) test data at all?
engine_ids = np.unique(test_windows.engine_ids)
n_engines_with_any_anomalous = 0
for eid in engine_ids:
    mask = test_windows.engine_ids == eid
    if y_true[mask].any():
        n_engines_with_any_anomalous += 1
print(f"\nEngines with >=1 anomalous window in observed test data: "
      f"{n_engines_with_any_anomalous} / {len(engine_ids)}")

# Life-fraction decile breakdown: raw score, detrended score, input
# variance, and true anomaly rate side by side.
total_life = compute_engine_total_life(test_df, test_rul)
life_frac = np.array([
    end_cycle / total_life.loc[eid]
    for eid, end_cycle in zip(test_windows.engine_ids, test_windows.end_cycles)
])
bins = np.linspace(0, 1.0, 11)
bin_idx = np.digitize(life_frac, bins) - 1

print("\nDecile | n     | score_raw | score_detrend | mean_input_var(detrend) | anom_rate")
for b in range(10):
    m = bin_idx == b
    if m.sum() > 0:
        print(
            f"  [{bins[b]:.1f}-{bins[b+1]:.1f}) | n={m.sum():5d} | "
            f"raw={scores_raw[m].mean():.4f} | detrend={scores_detrended[m].mean():.4f} | "
            f"var={input_variance_detrended[m].mean():.4f} | anom={y_true[m].mean():.2%}"
        )

# --- Decisive correlation check: is score just tracking variance? ---
corr_var_score_detrended, _ = spearmanr(input_variance_detrended, scores_detrended)
corr_var_score_raw, _ = spearmanr(input_variance_raw, scores_raw)
corr_var_label, _ = spearmanr(input_variance_detrended, y_true.astype(float))

print(f"\nSpearman corr(input_variance, score) -- detrended pass: {corr_var_score_detrended:.3f}")
print(f"Spearman corr(input_variance, score) -- raw pass:       {corr_var_score_raw:.3f}")
print(f"Spearman corr(input_variance[detrend], y_true):         {corr_var_label:.3f}")

print(
    "\nInterpretation guide:\n"
    "  - If corr(variance, score_detrended) is strongly positive (>0.7) and\n"
    "    corr(variance, y_true) is strongly NEGATIVE, the model is likely a\n"
    "    degenerate/near-mean-predictor: score tracks input variance, and\n"
    "    variance happens to be anti-correlated with the true label via\n"
    "    detrending -- fix training/architecture before touching scoring.\n"
    "  - If raw and detrended score patterns look similar (both invert),\n"
    "    the inversion predates detrending -- points at the AE itself.\n"
    "  - If raw scores separate healthy/anomalous correctly but detrended\n"
    "    scores don't, detrending is actively destroying the signal --\n"
    "    reconsider or remove it rather than adding delta-scoring on top."
)