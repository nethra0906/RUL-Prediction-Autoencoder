"""Healthy-region selection for autoencoder training.

See AI_CONTEXT.md Section 5.3 (healthy-region labels for autoencoder
training) and Section 19 (module contract).

Definition used for this checkpoint (fraction-based, matching the
closest prior-art paper's 85/15 split — arXiv 2601.10269): for each
engine trajectory, the first ``healthy_frac`` fraction of its observed
cycles is treated as healthy. This generalizes across engines of
different lifetimes better than a fixed cycle count, since C-MAPSS
engines fail at varying cycle counts.

This module only ever uses information within a single training
engine's own trajectory (its own final observed cycle) — it never
looks at another engine or at test-set information, so it does not
violate the leakage rules in AI_CONTEXT.md Section 17.
"""

from __future__ import annotations

import pandas as pd


def select_healthy_region(
    df: pd.DataFrame,
    healthy_frac: float = 0.85,
    engine_col: str = "unit_id",
    cycle_col: str = "cycle",
) -> pd.DataFrame:
    """Return only the healthy-region rows of each engine trajectory.

    For every engine (grouped by ``engine_col``), computes that
    engine's final observed cycle ``T`` and keeps rows with
    ``cycle <= healthy_frac * T``.

    Args:
        df: Training DataFrame with at least ``engine_col`` and
            ``cycle_col``. Should contain full run-to-failure
            trajectories (training data) — not truncated test data.
        healthy_frac: Fraction of each engine's life treated as
            healthy. Must be in (0, 1]. Always pass explicitly from
            experiment configuration (AI_CONTEXT.md Section 20) rather
            than relying on this default in production experiments.
        engine_col: Column identifying the engine/unit. Defaults to
            ``"unit_id"``.
        cycle_col: Column identifying the cycle index. Defaults to
            ``"cycle"``.

    Returns:
        A copy of the subset of ``df`` rows considered healthy.

    Raises:
        ValueError: If ``healthy_frac`` is not in (0, 1], or if
            ``engine_col`` / ``cycle_col`` are missing from ``df``.
    """
    if not (0.0 < healthy_frac <= 1.0):
        raise ValueError(f"healthy_frac must be in (0, 1], got {healthy_frac!r}")

    missing = {engine_col, cycle_col} - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")

    final_cycle = df.groupby(engine_col)[cycle_col].transform("max")
    cutoff = healthy_frac * final_cycle

    return df[df[cycle_col] <= cutoff].copy()


__all__ = ["select_healthy_region"]
