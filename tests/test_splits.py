import pandas as pd
import pytest

from src.data.splits import split_by_engine


def _make_synthetic_df(n_engines: int = 20, cycles_per_engine: int = 10) -> pd.DataFrame:
    rows = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, cycles_per_engine + 1):
            rows.append({"unit_id": engine_id, "cycle": cycle, "sensor_1": cycle * 0.1})
    return pd.DataFrame(rows)


def test_no_engine_overlap_between_train_and_val():
    df = _make_synthetic_df()
    train_df, val_df = split_by_engine(df, val_frac=0.2, seed=42)

    train_ids = set(train_df["unit_id"].unique())
    val_ids = set(val_df["unit_id"].unique())

    assert train_ids.isdisjoint(val_ids)


def test_all_engines_covered():
    df = _make_synthetic_df()
    train_df, val_df = split_by_engine(df, val_frac=0.2, seed=42)

    all_ids = set(df["unit_id"].unique())
    covered_ids = set(train_df["unit_id"].unique()) | set(val_df["unit_id"].unique())

    assert covered_ids == all_ids


def test_val_fraction_approximately_respected():
    df = _make_synthetic_df(n_engines=20)
    train_df, val_df = split_by_engine(df, val_frac=0.2, seed=42)

    n_val_engines = val_df["unit_id"].nunique()
    n_train_engines = train_df["unit_id"].nunique()

    assert n_val_engines == 4  # round(20 * 0.2)
    assert n_train_engines == 16


def test_deterministic_with_same_seed():
    df = _make_synthetic_df()
    train_a, val_a = split_by_engine(df, val_frac=0.2, seed=7)
    train_b, val_b = split_by_engine(df, val_frac=0.2, seed=7)

    assert set(train_a["unit_id"]) == set(train_b["unit_id"])
    assert set(val_a["unit_id"]) == set(val_b["unit_id"])


def test_different_seeds_can_differ():
    df = _make_synthetic_df(n_engines=50)
    train_a, val_a = split_by_engine(df, val_frac=0.2, seed=1)
    train_b, val_b = split_by_engine(df, val_frac=0.2, seed=2)

    assert set(val_a["unit_id"]) != set(val_b["unit_id"])


def test_invalid_val_frac_raises():
    df = _make_synthetic_df()
    with pytest.raises(ValueError):
        split_by_engine(df, val_frac=0.0)
    with pytest.raises(ValueError):
        split_by_engine(df, val_frac=1.0)


def test_missing_engine_col_raises():
    df = pd.DataFrame({"cycle": [1, 2, 3]})
    with pytest.raises(ValueError):
        split_by_engine(df, engine_col="unit_id")
