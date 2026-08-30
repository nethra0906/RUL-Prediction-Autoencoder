"""Fit/transform normalization, fit on training data only.

See AI_CONTEXT.md Section 19 (`data/normalization.py` module contract)
and Section 17 Rule 2 (fit scaler on training data only, transform
validation/test — never fit on data outside the allowed split).


"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NormalizationStats:
    """Per-feature mean/std fitted on training data.

    Attributes:
        feature_cols: Column names these stats apply to, in order.
        mean: Per-feature mean, shape (len(feature_cols),).
        std: Per-feature std, shape (len(feature_cols),). Any zero
            entries are replaced with 1.0 at fit time (see
            `fit_normalizer`) to avoid division by zero for constant
            sensors (AI_CONTEXT.md notes some sensors are
            near-constant, e.g. sensors 1, 5, 10, 16, 18, 19 per the
            closest prior-art paper's EDA).
    """

    feature_cols: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray


def fit_normalizer(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> NormalizationStats:
    """Fit per-feature mean/std on training data only.

    Args:
        train_df: Training DataFrame (or a healthy-region subset of
            it — see `data/healthy_region.py`) containing every column
            in `feature_cols`. Must never be validation or test data
            (AI_CONTEXT.md Section 17 Rule 2).
        feature_cols: Columns to compute normalization statistics for,
            in order (typically settings + sensors from
            `data/schema.py`).

    Returns:
        A `NormalizationStats` instance to be passed to `transform`.

    Raises:
        ValueError: If `train_df` is empty or any `feature_cols` entry
            is missing from it.
    """
    feature_cols = list(feature_cols)
    missing = set(feature_cols) - set(train_df.columns)
    if missing:
        raise ValueError(f"train_df is missing required column(s): {sorted(missing)}")
    if len(train_df) == 0:
        raise ValueError("train_df must be non-empty")

    values = train_df[feature_cols].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)  # avoid div-by-zero on constant sensors

    return NormalizationStats(feature_cols=tuple(feature_cols), mean=mean, std=std_safe)


def transform(df: pd.DataFrame, stats: NormalizationStats) -> pd.DataFrame:
    """Apply previously-fitted normalization stats to a DataFrame.

    Args:
        df: DataFrame containing every column in `stats.feature_cols`
            (train, validation, or test — the same fitted `stats`
            object is reused across all three, per AI_CONTEXT.md
            Section 17 Rule 2).
        stats: Output of `fit_normalizer`.

    Returns:
        A copy of `df` with `stats.feature_cols` replaced by their
        z-scored values. Other columns are left untouched.

    Raises:
        ValueError: If any `stats.feature_cols` entry is missing from
            `df`.
    """
    missing = set(stats.feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required column(s): {sorted(missing)}")

    result = df.copy()
    values = result[list(stats.feature_cols)].to_numpy(dtype=float)
    normalized = (values - stats.mean) / stats.std
    result[list(stats.feature_cols)] = normalized
    return result


__all__ = ["NormalizationStats", "fit_normalizer", "transform"]
