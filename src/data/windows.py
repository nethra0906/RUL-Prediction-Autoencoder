"""Sliding-window sequence construction, engine-boundary safe.

See AI_CONTEXT.md Section 19 (`data/windows.py` module contract) and
Section 17 Rule 1 (leakage — windows must never cross engine
boundaries; a window is only ever built from consecutive cycles of a
single engine's own trajectory).

Output preserves metadata alongside the windowed tensor so every
window can always be traced back to its engine and ending cycle (see
AI_CONTEXT.md Section 6.2, Aryaman's coding preferences: "Ensure each
window can be traced back to its engine and ending cycle").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data.schema import SENSOR_COLUMNS, SETTING_COLUMNS

_DEFAULT_FEATURE_COLUMNS = [*SETTING_COLUMNS, *SENSOR_COLUMNS]


@dataclass(frozen=True)
class WindowedSequences:
    """Structured output of `create_windows`.

    Attributes:
        X: Windowed feature tensor, shape (n_windows, window_size,
            n_features).
        engine_ids: Engine ID each window belongs to, shape (n_windows,).
        end_cycles: The cycle index of the *last* row in each window
            (i.e. the "as-of" cycle the window represents), shape
            (n_windows,).
        start_cycles: The cycle index of the *first* row in each
            window, shape (n_windows,).
        feature_names: Column names corresponding to the last axis of
            X, in order.
        y: Optional label value per window (see `label_cols` in
            `create_windows`), taken from the window's last row.
            Shape (n_windows, len(label_cols)) if requested, else None.
        label_names: Column names corresponding to the last axis of
            `y`, in order. None if `y` is None.
    """

    X: np.ndarray
    engine_ids: np.ndarray
    start_cycles: np.ndarray
    end_cycles: np.ndarray
    feature_names: list[str]
    y: Optional[np.ndarray] = None
    label_names: Optional[list[str]] = None

    def __len__(self) -> int:
        return self.X.shape[0]


def create_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int = 1,
    feature_cols: Optional[Sequence[str]] = None,
    label_cols: Optional[Sequence[str]] = None,
    engine_col: str = "unit_id",
    cycle_col: str = "cycle",
) -> WindowedSequences:
    """Build sliding-window sequences from a C-MAPSS-style DataFrame.

    For every engine (grouped by `engine_col`), rows are sorted by
    `cycle_col` and then sliced into overlapping windows of length
    `window_size`, stepping by `stride`. Windows are built exclusively
    from a single engine's own consecutive cycles — they never span
    two engines (AI_CONTEXT.md Section 17, Rule 1). An engine whose
    trajectory is shorter than `window_size` contributes zero windows
    and is silently skipped (not an error), since this is expected for
    short trajectories and should not halt processing of the rest of
    the dataset.

    Args:
        df: DataFrame containing at least `engine_col`, `cycle_col`,
            and every column in `feature_cols` (and `label_cols`, if
            given). Should already be restricted to the desired region
            (e.g. output of `select_healthy_region`) before windowing,
            if that's the intent — this function does not filter rows.
        window_size: Number of consecutive cycles per window. Must be
            a positive integer. Always pass explicitly from experiment
            configuration (AI_CONTEXT.md Section 20) rather than
            relying on a hard-coded value.
        stride: Step size between the start of consecutive windows
            within an engine. Must be a positive integer. Defaults to 1
            (maximally overlapping windows).
        feature_cols: Columns to include in the windowed tensor X, in
            order. Defaults to the 3 operational settings + 21 sensors
            (`SETTING_COLUMNS + SENSOR_COLUMNS` from `data/schema.py`).
            Pass explicitly to use a subset (e.g. sensors only).
        label_cols: Optional columns to also extract per window (taken
            from the window's *last* row — the "as-of" cycle). Useful
            for e.g. attaching a RUL label already computed by
            `data/rul.py`. If None, `WindowedSequences.y` is None.
        engine_col: Column identifying the engine/unit. Defaults to
            `"unit_id"`.
        cycle_col: Column identifying the cycle index within an engine
            trajectory. Defaults to `"cycle"`.

    Returns:
        A `WindowedSequences` instance. If no engine has enough cycles
        to produce even one window, all arrays are empty (shape[0]==0)
        rather than raising — callers should check `len(result) == 0`
        if that's a concern for their use case.

    Raises:
        ValueError: If `window_size` / `stride` are not positive
            integers, or if any required column is missing from `df`.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"window_size must be a positive integer, got {window_size!r}")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride!r}")

    feature_cols = list(feature_cols) if feature_cols is not None else list(_DEFAULT_FEATURE_COLUMNS)
    label_cols = list(label_cols) if label_cols is not None else None

    required = {engine_col, cycle_col, *feature_cols, *(label_cols or [])}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")

    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    engine_id_list: list = []
    start_cycle_list: list = []
    end_cycle_list: list = []

    for engine_id, group in df.groupby(engine_col, sort=False):
        group = group.sort_values(cycle_col)

        n_rows = len(group)
        if n_rows < window_size:
            # Trajectory too short to produce even one window — skip.
            continue

        feature_values = group[feature_cols].to_numpy()
        cycle_values = group[cycle_col].to_numpy()
        label_values = group[label_cols].to_numpy() if label_cols else None

        last_start = n_rows - window_size
        for start in range(0, last_start + 1, stride):
            end = start + window_size  # exclusive
            X_chunks.append(feature_values[start:end])
            start_cycle_list.append(cycle_values[start])
            end_cycle_list.append(cycle_values[end - 1])
            engine_id_list.append(engine_id)
            if label_values is not None:
                y_chunks.append(label_values[end - 1])

    if X_chunks:
        X = np.stack(X_chunks, axis=0)
        engine_ids = np.array(engine_id_list)
        start_cycles = np.array(start_cycle_list)
        end_cycles = np.array(end_cycle_list)
        y = np.stack(y_chunks, axis=0) if label_cols else None
    else:
        # No engine long enough to produce a window — return empty,
        # correctly-shaped arrays rather than raising, so callers can
        # compose this with e.g. per-FD looping without special-casing.
        n_features = len(feature_cols)
        X = np.empty((0, window_size, n_features))
        engine_ids = np.empty((0,))
        start_cycles = np.empty((0,))
        end_cycles = np.empty((0,))
        y = np.empty((0, len(label_cols))) if label_cols else None

    return WindowedSequences(
        X=X,
        engine_ids=engine_ids,
        start_cycles=start_cycles,
        end_cycles=end_cycles,
        feature_names=feature_cols,
        y=y,
        label_names=label_cols,
    )


__all__ = ["WindowedSequences", "create_windows"]