from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


DEFAULT_CONDITION_COLS = (
    "setting_1",
    "setting_2",
    "setting_3",
)


@dataclass(frozen=True)
class RegimeModel:
    """Fitted operating-regime model."""

    feature_cols: tuple[str, ...]
    n_regimes: int
    model: KMeans


def fit_regime_model(
    train_df: pd.DataFrame,
    condition_cols: Sequence[str] = DEFAULT_CONDITION_COLS,
    n_regimes: int = 6,
    seed: int = 42,
) -> RegimeModel:
    """Fit an operating-regime model using training data only."""

    condition_cols = tuple(condition_cols)

    missing = set(condition_cols) - set(train_df.columns)
    if missing:
        raise ValueError(
            f"train_df is missing required column(s): {sorted(missing)}"
        )

    if len(train_df) == 0:
        raise ValueError("train_df must be non-empty")

    values = train_df[list(condition_cols)].to_numpy(dtype=float)

    model = KMeans(
        n_clusters=n_regimes,
        random_state=seed,
        n_init=10,
    )
    model.fit(values)

    return RegimeModel(
        feature_cols=condition_cols,
        n_regimes=n_regimes,
        model=model,
    )


def assign_regimes(
    df: pd.DataFrame,
    regime_model: RegimeModel,
) -> pd.Series:
    """Assign each row to a fitted operating regime."""

    missing = set(regime_model.feature_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"df is missing required column(s): {sorted(missing)}"
        )

    values = df[list(regime_model.feature_cols)].to_numpy(dtype=float)

    labels = regime_model.model.predict(values)

    return pd.Series(
        labels,
        index=df.index,
        name="operating_regime",
        dtype="int64",
    )


__all__ = [
    "DEFAULT_CONDITION_COLS",
    "RegimeModel",
    "fit_regime_model",
    "assign_regimes",
]