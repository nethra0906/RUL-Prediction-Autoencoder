"""Piecewise-linear / capped RUL label generation.

See AI_CONTEXT.md Section 5 (RUL labeling) and Section 18/19
(`data/rul.py` module contract).

For a training engine with final observed cycle ``T``, the uncapped RUL
at cycle ``t`` is ``T - t``. Because early-life degradation is largely
indistinguishable from healthy operation, the community-standard
"piecewise-linear" / capped formulation clips this to a maximum value:

    RUL_capped(t) = min(T - t, RUL_MAX)

``RUL_MAX`` is a modeling choice, not a fixed constant, and must always
be supplied by the caller (via function argument / experiment config)
rather than hard-coded here — see AI_CONTEXT.md Section 5.1 and
Section 20 (configuration philosophy).

Leakage note (AI_CONTEXT.md Section 17, Rule 5): this module only ever
uses information available within the trajectory it is labeling (the
final observed cycle of a *training* engine, which runs to failure by
construction). It must never be used to inject test-set failure
information into model input or threshold selection.
"""

from __future__ import annotations

import pandas as pd


def generate_capped_rul(
    df: pd.DataFrame,
    max_rul: int,
    engine_col: str = "unit_id",
    cycle_col: str = "cycle",
    output_col: str = "RUL",
) -> pd.DataFrame:
    """Compute piecewise-linear (capped) RUL labels for each row.

    For every engine trajectory in ``df`` (grouped by ``engine_col``),
    the final observed cycle is treated as the failure cycle ``T``. Each
    row's raw RUL is ``T - cycle``, which is then capped from above at
    ``max_rul``:

        RUL_capped(t) = min(T - t, max_rul)

    This assumes each engine's trajectory in ``df`` already runs to
    failure (i.e. this is training data, not truncated test data — see
    AI_CONTEXT.md Section 5.2 for why test RUL must come from the
    provided ground-truth file instead).

    Args:
        df: DataFrame containing at least ``engine_col`` and
            ``cycle_col``. Not mutated; a copy with the new RUL column
            is returned.
        max_rul: Upper cap applied to the RUL label. Must be a positive
            integer. Always pass this explicitly from experiment
            configuration — never hard-code it at the call site.
        engine_col: Column identifying the engine/unit. Defaults to
            ``"unit_id"`` per the canonical C-MAPSS schema
            (see `data/schema.py`).
        cycle_col: Column identifying the cycle index within an engine
            trajectory. Defaults to ``"cycle"``.
        output_col: Name of the RUL column to add. Defaults to
            ``"RUL"``.

    Returns:
        A copy of ``df`` with an additional ``output_col`` column
        containing the capped RUL for every row.

    Raises:
        ValueError: If ``max_rul`` is not a positive integer, or if
            ``engine_col`` / ``cycle_col`` are missing from ``df``.
    """
    if not isinstance(max_rul, int) or max_rul <= 0:
        raise ValueError(f"max_rul must be a positive integer, got {max_rul!r}")

    missing = {engine_col, cycle_col} - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")

    result = df.copy()

    # Failure cycle per engine = last observed cycle of that trajectory.
    failure_cycle = result.groupby(engine_col)[cycle_col].transform("max")

    raw_rul = failure_cycle - result[cycle_col]
    result[output_col] = raw_rul.clip(upper=max_rul)

    return result


__all__ = ["generate_capped_rul"]
