"""One-off diagnostic: inspect test-set score/label separation for the
baseline run, to debug why ROC-AUC is below 0.5 (see run_baseline.py).

Not part of the permanent pipeline — a throwaway investigation script.
Run from repo root: `python scripts/diagnose_test_scores.py`
"""

import numpy as np
import torch

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
from src.anomaly.reconstruction import window_scores_numpy

FEATURE_COLS = [*SETTING_COLUMNS, *SENSOR_COLUMNS]
FD_ID = "FD001"
WINDOW_SIZE = 30
STRIDE = 1
HEALTHY_FRAC = 0.85

# Rebuild the same train-fitted normalization stats used by run_baseline.
train_full = load_train(FD_ID)
train_split, _ = split_by_engine(train_full, val_frac=0.15, seed=42)
train_healthy = select_healthy_region(train_split, healthy_frac=HEALTHY_FRAC)
norm_stats = fit_normalizer(train_healthy, feature_cols=FEATURE_COLS)

# Reload the trained model checkpoint (from the 100-epoch run).
model = DenseAutoencoder(window_size=WINDOW_SIZE, n_features=len(FEATURE_COLS), latent_dim=16, hidden_dims=(64, 32))
model.load_state_dict(torch.load("results/checkpoints/fd001_ae_fixed_v001.pt"))
model.eval()

# Rebuild test windows + labels exactly as run_baseline does.
test_df = load_test(FD_ID)
test_rul = load_test_rul(FD_ID)
test_labeled = label_anomalous_by_life_fraction(test_df, test_rul, healthy_frac=HEALTHY_FRAC)
test_norm = transform(test_labeled, norm_stats)
test_windows = create_windows(
    test_norm, window_size=WINDOW_SIZE, stride=STRIDE, feature_cols=FEATURE_COLS, label_cols=["is_anomalous"]
)

test_X = torch.tensor(test_windows.X, dtype=torch.float32)
with torch.no_grad():
    recon, _ = model(test_X)
scores = window_scores_numpy(test_X, recon)
y_true = test_windows.y.flatten().astype(bool)

print(f"Total test windows: {len(scores)}")
print(f"Anomalous windows: {y_true.sum()} ({y_true.mean():.2%})")
print()
print("Score stats (healthy windows):")
print(f"  mean={scores[~y_true].mean():.4f}  std={scores[~y_true].std():.4f}  "
      f"min={scores[~y_true].min():.4f}  max={scores[~y_true].max():.4f}")
print("Score stats (anomalous windows):")
if y_true.sum() > 0:
    print(f"  mean={scores[y_true].mean():.4f}  std={scores[y_true].std():.4f}  "
          f"min={scores[y_true].min():.4f}  max={scores[y_true].max():.4f}")
else:
    print("  (none)")
print()

# How many engines actually reach the anomalous region within their
# OBSERVED (truncated) test data at all?
engine_ids = np.unique(test_windows.engine_ids)
n_engines_with_any_anomalous = 0
for eid in engine_ids:
    mask = test_windows.engine_ids == eid
    if y_true[mask].any():
        n_engines_with_any_anomalous += 1
print(f"Engines with >=1 anomalous window in observed test data: "
      f"{n_engines_with_any_anomalous} / {len(engine_ids)}")

# Score trend over normalized life-fraction, pooled across engines, to see
# whether reconstruction error rises or falls as engines approach failure.
total_life = compute_engine_total_life(test_df, test_rul)
life_frac = np.array([
    end_cycle / total_life.loc[eid]
    for eid, end_cycle in zip(test_windows.engine_ids, test_windows.end_cycles)
])
bins = np.linspace(0, 1.0, 11)
bin_idx = np.digitize(life_frac, bins) - 1
print()
print("Mean reconstruction error by life-fraction decile (0=start of life, 9=near/at failure):")
for b in range(10):
    bin_scores = scores[bin_idx == b]
    if len(bin_scores) > 0:
        print(f"  [{bins[b]:.1f}-{bins[b+1]:.1f}): n={len(bin_scores):5d}  mean_score={bin_scores.mean():.4f}")