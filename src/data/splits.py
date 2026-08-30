"""Engine-level train/validation splitting.

See AI_CONTEXT.md Section 17 Rule 1 (split by engine, not random rows)
and Section 19 (`data/splits.py` module contract).

Project convention (confirmed for this checkpoint): C-MAPSS already
supplies a separate official test file (`test_FDx.txt` + `RUL_FDx.txt`),
which is used as the held-out test set as-is. This module only splits
the *training* file's engines into a train/validation partition —
there is no three-way train/val/test split here.

Leakage rule (non-negotiable, AI_CONTEXT.md Section 17 Rule 1): no
engine's cycles may appear in both the train and validation partitions.
Splitting is done by shuffling whole engine IDs, never by row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_engine(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    engine_col: str = "unit_id",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training DataFrame into train/validation by engine ID.

    Args:
        df: DataFrame containing at least ``engine_col``. Every row for
            a given engine stays together on one side of the split.
        val_frac: Fraction of *engines* (not rows) assigned to
            validation. Must be in (0, 1).
        engine_col: Column identifying the engine/unit. Defaults to
            ``"unit_id"`` per the canonical C-MAPSS schema (see
            `data/schema.py`).
        seed: Random seed for the engine shuffle. Always pass this
            explicitly from experiment configuration for reproducibility
            (AI_CONTEXT.md Section 20/21).

    Returns:
        ``(train_df, val_df)`` — both are copies of the relevant rows of
        ``df``, with no engine appearing in both.

    Raises:
        ValueError: If ``val_frac`` is not in (0, 1), if ``engine_col``
            is missing, or if there are too few engines to produce a
            non-empty split on both sides.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac!r}")

    if engine_col not in df.columns:
        raise ValueError(f"df is missing required column: {engine_col!r}")

    engine_ids = df[engine_col].unique()
    n_engines = len(engine_ids)

    n_val = round(n_engines * val_frac)
    if n_val == 0 or n_val == n_engines:
        raise ValueError(
            f"val_frac={val_frac!r} yields an empty split with "
            f"{n_engines} engines (n_val={n_val}). Use a larger dataset "
            "or a less extreme val_frac."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(engine_ids)

    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])

    # Leakage assertion: partitions must be disjoint and complete.
    assert train_ids.isdisjoint(val_ids), "train/val engine sets overlap"
    assert train_ids | val_ids == set(engine_ids), "engine sets don't cover all engines"

    train_df = df[df[engine_col].isin(train_ids)].copy()
    val_df = df[df[engine_col].isin(val_ids)].copy()

    return train_df, val_df


__all__ = ["split_by_engine"]
