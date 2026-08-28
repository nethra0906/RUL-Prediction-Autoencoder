"""Unit tests for `src/data/rul.py`.

See AI_CONTEXT.md Section 5 (RUL labeling) and Section 18 (`tests/`
contract: every important mathematical component should have tests).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.rul import generate_capped_rul


def _toy_trajectory_df() -> pd.DataFrame:
    """Two engines: unit 1 runs 5 cycles, unit 2 runs 3 cycles."""
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 5, 1, 2, 3],
        }
    )


def test_uncapped_region_matches_linear_countdown():
    df = _toy_trajectory_df()
    out = generate_capped_rul(df, max_rul=125)

    unit1 = out[out["unit_id"] == 1].sort_values("cycle")
    assert unit1["RUL"].tolist() == [4, 3, 2, 1, 0]

    unit2 = out[out["unit_id"] == 2].sort_values("cycle")
    assert unit2["RUL"].tolist() == [2, 1, 0]


def test_rul_is_capped_at_max_rul():
    df = pd.DataFrame({"unit_id": [1] * 10, "cycle": list(range(1, 11))})
    out = generate_capped_rul(df, max_rul=5)

    # Failure cycle = 10, so raw RUL at cycle 1 would be 9 -> capped to 5.
    assert out.loc[out["cycle"] == 1, "RUL"].item() == 5
    assert out.loc[out["cycle"] == 10, "RUL"].item() == 0
    assert out["RUL"].max() <= 5


def test_last_cycle_always_has_zero_rul():
    df = _toy_trajectory_df()
    out = generate_capped_rul(df, max_rul=125)

    for unit_id, group in out.groupby("unit_id"):
        last_cycle_row = group.loc[group["cycle"].idxmax()]
        assert last_cycle_row["RUL"] == 0


def test_engines_are_handled_independently():
    # Unit 2 is shorter than unit 1; its RUL countdown must not be
    # influenced by unit 1's trajectory length (no cross-engine leakage).
    df = _toy_trajectory_df()
    out = generate_capped_rul(df, max_rul=125)

    unit2_first_cycle_rul = out.loc[
        (out["unit_id"] == 2) & (out["cycle"] == 1), "RUL"
    ].item()
    assert unit2_first_cycle_rul == 2  # NOT influenced by unit 1's length=5


def test_does_not_mutate_input_df():
    df = _toy_trajectory_df()
    original_columns = list(df.columns)
    generate_capped_rul(df, max_rul=125)
    assert list(df.columns) == original_columns
    assert "RUL" not in df.columns


def test_custom_column_names():
    df = pd.DataFrame(
        {
            "engine": [1, 1, 1],
            "t": [1, 2, 3],
        }
    )
    out = generate_capped_rul(
        df, max_rul=125, engine_col="engine", cycle_col="t", output_col="rul_label"
    )
    assert out["rul_label"].tolist() == [2, 1, 0]


@pytest.mark.parametrize("bad_max_rul", [0, -1, 12.5, "125"])
def test_invalid_max_rul_raises(bad_max_rul):
    df = _toy_trajectory_df()
    with pytest.raises(ValueError):
        generate_capped_rul(df, max_rul=bad_max_rul)


def test_missing_required_column_raises():
    df = pd.DataFrame({"unit_id": [1, 1], "not_cycle": [1, 2]})
    with pytest.raises(ValueError):
        generate_capped_rul(df, max_rul=125)
