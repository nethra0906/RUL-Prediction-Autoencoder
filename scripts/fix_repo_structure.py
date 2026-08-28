"""Repository repair script.

Fixes a terminal-creation error where `touch src/data/{a.py,b.py,...}` (or
its PowerShell equivalent) was run without brace expansion, producing a
single literal file whose name contains the whole `{...}` list instead of
the individual `.py` files it was meant to create.

This script:
  1. Finds every literal file whose name contains `{` and `}` inside the
     known `src/` subpackages.
  2. Parses the comma-separated names out of the braces.
  3. Creates each individual `.py` file (with a minimal module docstring
     stub) if it does not already exist.
  4. Deletes the malformed literal file.

Safe to re-run: it never overwrites a `.py` file that already exists with
real content, and it only touches files whose name literally contains
`{` or `}`.

Usage (from repo root):
    python scripts/fix_repo_structure.py
"""

from __future__ import annotations

import re
from pathlib import Path

# Directories known to contain a malformed `{a.py,b.py,...}` file per the
# AI_CONTEXT.md Section 18 repository layout.
TARGET_DIRS = [
    "src/data",
    "src/anomaly",
    "src/evaluation",
    "src/models",
    "src/experiments",
    "src/training",
]

# One-line responsibility stubs used to seed each new module's docstring.
# Falls back to a generic stub for any filename not listed here.
MODULE_STUBS: dict[str, str] = {
    "loaders.py": "Read raw C-MAPSS text files into standardized, engine/cycle-indexed DataFrames.",
    "schema.py": "Canonical C-MAPSS column schema.",
    "splits.py": "Engine-level train/validation/test splitting (leakage-free).",
    "rul.py": "Piecewise-linear / capped RUL label generation.",
    "healthy_region.py": "Healthy-region cycle selection for autoencoder training.",
    "normalization.py": "Fit/transform normalization, fit on training data only.",
    "windows.py": "Sliding-window sequence construction, engine-boundary safe.",
    "autoencoder.py": "Baseline (vanilla) autoencoder architecture.",
    "regime_encoder.py": "Maps the three operational settings to a regime embedding.",
    "conditioned_autoencoder.py": "Regime-conditioned autoencoder (sensor + regime representation).",
    "rul_head.py": "Optional joint RUL regression head sharing the AE latent space.",
    "reconstruction.py": "Total / per-time-step / per-channel reconstruction error.",
    "fixed_threshold.py": "Baseline fixed reconstruction-error threshold detector.",
    "dynamic_threshold.py": "Heuristic dynamic threshold (precursor to conformal calibration).",
    "conformal.py": "Conformal calibration: nonconformity scores, quantile, dynamic threshold.",
    "event_detection.py": "Persistence / consecutive-alert rule for event-level detection.",
    "attribution.py": "Per-sensor reconstruction-error attribution and ranking.",
    "train_autoencoder.py": "Training loop for the (baseline or conditioned) autoencoder.",
    "train_joint.py": "Training loop for the joint reconstruction + RUL model (stretch goal).",
    "losses.py": "Loss functions, including the asymmetric RUL loss (stretch goal).",
    "callbacks.py": "Training callbacks (early stopping, checkpointing, logging).",
    "classification_metrics.py": "Precision, Recall, F1, ROC-AUC, false-alarm rate.",
    "rul_metrics.py": "MAE and RMSE for RUL predictions.",
    "nasa_score.py": "Asymmetric NASA scoring function for RUL predictions.",
    "lead_time.py": "Detection lead-time calculation (failure cycle - first valid alert cycle).",
    "evaluation_runner.py": "Orchestrates a full evaluation run and result-table assembly.",
    "run_baseline.py": "Experiment entry point: fixed-threshold baseline autoencoder.",
    "run_dynamic_threshold.py": "Experiment entry point: dynamic-threshold precursor.",
    "run_regime_conditioned.py": "Experiment entry point: regime-conditioned autoencoder.",
    "run_conformal.py": "Experiment entry point: conformal-calibrated regime-conditioned model.",
}

BRACE_FILE_RE = re.compile(r"^\{(.*)\}$")


def stub_contents(filename: str) -> str:
    summary = MODULE_STUBS.get(filename, "TODO: implement this module.")
    return f'"""{summary}\n\nSee AI_CONTEXT.md Section 18/19 for this module\'s contract.\n"""\n'


def fix_directory(dir_path: Path) -> None:
    if not dir_path.is_dir():
        print(f"  [skip] {dir_path} does not exist")
        return

    for entry in dir_path.iterdir():
        if not entry.is_file():
            continue
        if "{" not in entry.name or "}" not in entry.name:
            continue

        match = BRACE_FILE_RE.match(entry.name)
        inner = match.group(1) if match else entry.name.strip("{}")
        names = [n.strip() for n in inner.split(",") if n.strip()]

        print(f"  Found malformed file: {entry.name}")
        for name in names:
            target = dir_path / name
            if target.exists():
                print(f"    [keep]   {target} already exists")
                continue
            target.write_text(stub_contents(name), encoding="utf-8")
            print(f"    [create] {target}")

        entry.unlink()
        print(f"    [delete] {entry}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Repairing repository structure under: {repo_root}")
    for rel_dir in TARGET_DIRS:
        dir_path = repo_root / rel_dir
        print(f"Checking {rel_dir} ...")
        fix_directory(dir_path)
    print("Done.")


if __name__ == "__main__":
    main()
