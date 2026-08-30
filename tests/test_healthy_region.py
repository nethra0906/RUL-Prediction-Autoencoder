import pandas as pd
import pytest

from src.data.healthy_region import select_healthy_region


def _make_engine_df(final_cycle: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1] * final_cycle,
            "cycle": list(range(1, final_cycle + 1)),
        }
    )


def test_keeps_only_cycles_up_to_fraction():
    df = _make_engine_df(final_cycle=100)
    healthy = select_healthy_region(df, healthy_frac=0.85)

    assert healthy["cycle"].max() == 85
    assert len(healthy) == 85


def test_multiple_engines_use_own_final_cycle():
    df = pd.concat(
        [
            _make_engine_df(100).assign(unit_id=1),
            _make_engine_df(200).assign(unit_id=2),
        ],
        ignore_index=True,
    )
    healthy = select_healthy_region(df, healthy_frac=0.5)

    engine_1_max = healthy[healthy["unit_id"] == 1]["cycle"].max()
    engine_2_max = healthy[healthy["unit_id"] == 2]["cycle"].max()

    assert engine_1_max == 50
    assert engine_2_max == 100


def test_healthy_frac_one_keeps_everything():
    df = _make_engine_df(final_cycle=50)
    healthy = select_healthy_region(df, healthy_frac=1.0)

    assert len(healthy) == len(df)


def test_invalid_healthy_frac_raises():
    df = _make_engine_df(final_cycle=50)
    with pytest.raises(ValueError):
        select_healthy_region(df, healthy_frac=0.0)
    with pytest.raises(ValueError):
        select_healthy_region(df, healthy_frac=1.5)


def test_missing_columns_raises():
    df = pd.DataFrame({"unit_id": [1, 2, 3]})
    with pytest.raises(ValueError):
        select_healthy_region(df)
