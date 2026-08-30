"""C-MAPSS raw file loaders.

See AI_CONTEXT.md Section 4 (file conventions) and Section 19
(`data/loaders.py` module contract).

Responsibilities (per AI_CONTEXT.md):
    - read C-MAPSS files,
    - return standardized pandas DataFrames,
    - preserve engine/cycle identifiers.

Known raw-file quirks handled here (AI_CONTEXT.md Section 4 explicitly
warns these must be inspected, not assumed — this loader has NOT yet
been run against real downloaded files; verify shape/columns/NaNs on
first use and adjust if the assumption below doesn't hold):
    - space-delimited, no header row
    - a trailing separator on each line often produces 1-2 extra
      all-NaN columns beyond the 26 real ones; these are dropped
    - `RUL_FDx.txt` is a single-column file, one integer per test engine,
      in the same engine order as the test file's unit_id 1..N

This module never modifies files under data/raw/ (AI_CONTEXT.md
Section 18, "data/raw" rule) — it only reads them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import COLUMN_NAMES

# Real C-MAPSS columns: unit_id, cycle, 3 settings, 21 sensors.
_N_REAL_COLUMNS = len(COLUMN_NAMES)


def _default_raw_dir() -> Path:
    """Project-relative default: data/raw/CMAPSS."""
    # src/data/loaders.py -> repo root is 2 parents up (src/, then root).
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "raw" / "CMAPSS"


def _resolve_fd_dir(fd_id: str, raw_dir: str | Path | None) -> Path:
    base = Path(raw_dir) if raw_dir is not None else _default_raw_dir()
    return base / fd_id


def _read_space_delimited(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited C-MAPSS file with no header.

    Trims any fully-empty trailing columns caused by a trailing
    delimiter on each line, then asserts the remaining column count
    matches the canonical schema before assigning names explicitly
    (never rely on positional/inferred columns per AI_CONTEXT.md
    Section 4).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected C-MAPSS file not found: {path}. "
            "Raw data is gitignored — place downloaded files under "
            "data/raw/CMAPSS/<FDxxx>/ locally (see README.md)."
        )

    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    # Drop fully-empty trailing columns from a trailing delimiter.
    raw = raw.dropna(axis=1, how="all")

    if raw.shape[1] != _N_REAL_COLUMNS:
        raise ValueError(
            f"{path}: expected {_N_REAL_COLUMNS} columns after dropping "
            f"empty trailing columns, got {raw.shape[1]}. Inspect the raw "
            "file manually (df.shape, df.head()) — the trailing-column "
            "assumption in this loader may not hold for this download."
        )

    raw.columns = COLUMN_NAMES
    return raw


def load_train(fd_id: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load a C-MAPSS training file (full run-to-failure trajectories).

    Args:
        fd_id: Sub-dataset identifier, e.g. "FD001".
        raw_dir: Optional override for the CMAPSS root directory.
            Defaults to <repo_root>/data/raw/CMAPSS.

    Returns:
        DataFrame with canonical columns from
        `src.data.schema.COLUMN_NAMES`.
    """
    fd_dir = _resolve_fd_dir(fd_id, raw_dir)
    return _read_space_delimited(fd_dir / f"train_{fd_id}.txt")


def load_test(fd_id: str, raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Load a C-MAPSS test file (truncated trajectories).

    See AI_CONTEXT.md Section 5.2: this file alone does NOT contain the
    true RUL at truncation — pair it with `load_test_rul` for evaluation.
    """
    fd_dir = _resolve_fd_dir(fd_id, raw_dir)
    return _read_space_delimited(fd_dir / f"test_{fd_id}.txt")


def load_test_rul(fd_id: str, raw_dir: str | Path | None = None) -> pd.Series:
    """Load ground-truth test RUL values.

    Returns a Series indexed 1..N (matching test-file unit_id order),
    named "RUL". Evaluation-only per AI_CONTEXT.md Section 17, Rule 5 —
    must never be used as model input or for threshold selection.
    """
    fd_dir = _resolve_fd_dir(fd_id, raw_dir)
    path = fd_dir / f"RUL_{fd_id}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Expected C-MAPSS file not found: {path}. "
            "Raw data is gitignored — place downloaded files under "
            "data/raw/CMAPSS/<FDxxx>/ locally (see README.md)."
        )

    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    raw = raw.dropna(axis=1, how="all")

    if raw.shape[1] != 1:
        raise ValueError(
            f"{path}: expected a single RUL column, got {raw.shape[1]}. "
            "Inspect the raw file manually."
        )

    series = raw.iloc[:, 0].astype(int)
    series.index = pd.RangeIndex(start=1, stop=len(series) + 1, name="unit_id")
    series.name = "RUL"
    return series


__all__ = ["load_train", "load_test", "load_test_rul"]
