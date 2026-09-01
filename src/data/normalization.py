"""Normalization utilities for CMAPSS preprocessing.

Supports:
1. Global z-score normalization.
2. Regime-aware z-score normalization.

Regime-aware normalization computes feature statistics separately
for each operating regime. The statistics are always fitted on
training data only and reused unchanged for validation/test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NormalizationStats:
    """Global per-feature normalization statistics."""

    feature_cols: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class RegimeNormalizationStats:
    """Per-feature normalization statistics for each operating regime."""

    feature_cols: tuple[str, ...]
    regime_col: str
    regimes: tuple[int, ...]
    means: dict[int, np.ndarray]
    stds: dict[int, np.ndarray]


def fit_normalizer(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> NormalizationStats:
    """Fit global per-feature z-score statistics on training data only."""

    feature_cols = list(feature_cols)

    missing = set(feature_cols) - set(train_df.columns)
    if missing:
        raise ValueError(
            f"train_df is missing required column(s): {sorted(missing)}"
        )

    if len(train_df) == 0:
        raise ValueError("train_df must be non-empty")

    values = train_df[feature_cols].to_numpy(dtype=float)

    mean = values.mean(axis=0)
    std = values.std(axis=0)

    std_safe = np.where(std == 0.0, 1.0, std)

    return NormalizationStats(
        feature_cols=tuple(feature_cols),
        mean=mean,
        std=std_safe,
    )


def transform(
    df: pd.DataFrame,
    stats: NormalizationStats,
) -> pd.DataFrame:
    """Apply previously fitted global normalization statistics."""

    missing = set(stats.feature_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"df is missing required column(s): {sorted(missing)}"
        )

    result = df.copy()

    values = result[list(stats.feature_cols)].to_numpy(dtype=float)
    normalized = (values - stats.mean) / stats.std

    result[list(stats.feature_cols)] = normalized

    return result


def fit_regime_normalizer(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    regime_col: str = "operating_regime",
) -> RegimeNormalizationStats:
    """Fit per-regime z-score statistics on training data only.

    Each operating regime receives its own mean and standard deviation
    for every feature.

    The DataFrame must already contain operating-regime labels produced
    by a regime model fitted on training data.
    """

    feature_cols = list(feature_cols)

    required = set(feature_cols) | {regime_col}
    missing = required - set(train_df.columns)

    if missing:
        raise ValueError(
            f"train_df is missing required column(s): {sorted(missing)}"
        )

    if len(train_df) == 0:
        raise ValueError("train_df must be non-empty")

    regimes = tuple(
        sorted(
            int(x)
            for x in train_df[regime_col].dropna().unique()
        )
    )

    if not regimes:
        raise ValueError("train_df contains no operating regimes")

    means: dict[int, np.ndarray] = {}
    stds: dict[int, np.ndarray] = {}

    for regime in regimes:
        regime_df = train_df[train_df[regime_col] == regime]

        if len(regime_df) == 0:
            raise ValueError(
                f"No training rows found for regime {regime}"
            )

        values = regime_df[feature_cols].to_numpy(dtype=float)

        mean = values.mean(axis=0)
        std = values.std(axis=0)

        std_safe = np.where(std == 0.0, 1.0, std)

        means[regime] = mean
        stds[regime] = std_safe

    return RegimeNormalizationStats(
        feature_cols=tuple(feature_cols),
        regime_col=regime_col,
        regimes=regimes,
        means=means,
        stds=stds,
    )


def transform_by_regime(
    df: pd.DataFrame,
    stats: RegimeNormalizationStats,
) -> pd.DataFrame:
    """Apply previously fitted per-regime normalization statistics."""

    required = set(stats.feature_cols) | {stats.regime_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"df is missing required column(s): {sorted(missing)}"
        )

    result = df.copy()

    unknown_regimes = set(
        int(x)
        for x in result[stats.regime_col].dropna().unique()
    ) - set(stats.regimes)

    if unknown_regimes:
        raise ValueError(
            "Encountered regime(s) not present in fitted training "
            f"statistics: {sorted(unknown_regimes)}"
        )

    for regime in stats.regimes:
        mask = result[stats.regime_col] == regime

        if not mask.any():
            continue

        values = result.loc[
            mask, list(stats.feature_cols)
        ].to_numpy(dtype=float)

        normalized = (
            values - stats.means[regime]
        ) / stats.stds[regime]

        result.loc[
            mask, list(stats.feature_cols)
        ] = normalized

    return result


__all__ = [
    "NormalizationStats",
    "RegimeNormalizationStats",
    "fit_normalizer",
    "transform",
    "fit_regime_normalizer",
    "transform_by_regime",
]