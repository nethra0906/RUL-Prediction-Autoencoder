"""Tests for `src/data/windows.py`.

See AI_CONTEXT.md Section 17 Rule 1 (no window may cross an engine
boundary) and Section 19 (`data/windows.py` contract: preserve engine
ID / start / end cycle metadata alongside every window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.windows import create_windows


def _make_engine_df(engine_id: int, n_cycles: int, feature_val_offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [engine_id] * n_cycles,
            "cycle": list(range(1, n_cycles + 1)),
            "sensor_1": [feature_val_offset + c for c in range(1, n_cycles + 1)],
            "sensor_2": [feature_val_offset - c for c in range(1, n_cycles + 1)],
        }
    )


def test_basic_window_shape_and_count():
    df = _make_engine_df(engine_id=1, n_cycles=10)
    result = create_windows(
        df, window_size=3, stride=1, feature_cols=["sensor_1", "sensor_2"]
    )

    # 10 cycles, window=3, stride=1 -> 8 windows (starts 0..7)
    assert result.X.shape == (8, 3, 2)
    assert len(result) == 8


def test_window_never_crosses_engine_boundary():
    df = pd.concat(
        [_make_engine_df(1, n_cycles=5), _make_engine_df(2, n_cycles=5, feature_val_offset=1000)],
        ignore_index=True,
    )
    result = create_windows(df, window_size=3, feature_cols=["sensor_1", "sensor_2"])

    # Each window's engine_id must be consistent with its feature values'
    # magnitude (engine 2 values are offset by 1000).
    for i in range(len(result)):
        eng = result.engine_ids[i]
        window_vals = result.X[i, :, 0]  # sensor_1 column
        if eng == 1:
            assert np.all(window_vals < 1000)
        else:
            assert np.all(window_vals >= 1000)


def test_engine_shorter_than_window_is_skipped():
    df = pd.concat(
        [_make_engine_df(1, n_cycles=2), _make_engine_df(2, n_cycles=10)],
        ignore_index=True,
    )
    result = create_windows(df, window_size=5, feature_cols=["sensor_1", "sensor_2"])

    # Engine 1 (only 2 cycles) contributes zero windows.
    assert np.all(result.engine_ids == 2)
    assert len(result) == 6  # 10 cycles, window=5, stride=1 -> 6 windows


def test_stride_reduces_window_count():
    df = _make_engine_df(engine_id=1, n_cycles=10)
    result_stride1 = create_windows(df, window_size=3, stride=1, feature_cols=["sensor_1"])
    result_stride2 = create_windows(df, window_size=3, stride=2, feature_cols=["sensor_1"])

    assert len(result_stride1) == 8
    assert len(result_stride2) == 4  # starts 0, 2, 4, 6


def test_start_and_end_cycles_are_correct():
    df = _make_engine_df(engine_id=1, n_cycles=5)
    result = create_windows(df, window_size=3, stride=1, feature_cols=["sensor_1"])

    # windows: [1,2,3], [2,3,4], [3,4,5]
    assert result.start_cycles.tolist() == [1, 2, 3]
    assert result.end_cycles.tolist() == [3, 4, 5]


def test_label_cols_extracted_from_last_row():
    df = _make_engine_df(engine_id=1, n_cycles=5)
    df["RUL"] = [40, 30, 20, 10, 0]
    result = create_windows(
        df, window_size=3, stride=1, feature_cols=["sensor_1"], label_cols=["RUL"]
    )

    # window ending at cycle 3 -> RUL=20; ending at 4 -> RUL=10; ending at 5 -> RUL=0
    assert result.y.flatten().tolist() == [20, 10, 0]
    assert result.label_names == ["RUL"]


def test_no_windows_returns_empty_not_error():
    df = _make_engine_df(engine_id=1, n_cycles=2)
    result = create_windows(df, window_size=10, feature_cols=["sensor_1", "sensor_2"])

    assert len(result) == 0
    assert result.X.shape == (0, 10, 2)


def test_default_feature_columns_match_schema():
    from src.data.schema import SENSOR_COLUMNS, SETTING_COLUMNS

    rows = {"unit_id": [1] * 5, "cycle": list(range(1, 6))}
    for col in SETTING_COLUMNS + SENSOR_COLUMNS:
        rows[col] = [0.0] * 5
    df = pd.DataFrame(rows)

    result = create_windows(df, window_size=3)
    assert result.feature_names == SETTING_COLUMNS + SENSOR_COLUMNS
    assert result.X.shape == (3, 3, 24)


def test_invalid_window_size_raises():
    df = _make_engine_df(engine_id=1, n_cycles=5)
    with pytest.raises(ValueError):
        create_windows(df, window_size=0, feature_cols=["sensor_1"])
    with pytest.raises(ValueError):
        create_windows(df, window_size=-3, feature_cols=["sensor_1"])


def test_invalid_stride_raises():
    df = _make_engine_df(engine_id=1, n_cycles=5)
    with pytest.raises(ValueError):
        create_windows(df, window_size=3, stride=0, feature_cols=["sensor_1"])


def test_missing_feature_column_raises():
    df = _make_engine_df(engine_id=1, n_cycles=5)
    with pytest.raises(ValueError):
        create_windows(df, window_size=3, feature_cols=["sensor_1", "nonexistent_col"])