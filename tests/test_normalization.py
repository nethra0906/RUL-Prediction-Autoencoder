import numpy as np
import pandas as pd
import pytest

from src.data.normalization import (
    fit_normalizer,
    transform,
    fit_regime_normalizer,
    transform_by_regime,
)


def test_global_normalizer_fits_mean_and_std():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
        }
    )

    stats = fit_normalizer(df, ["x", "y"])

    assert np.allclose(stats.mean, [2.0, 20.0])
    assert np.allclose(
        stats.std,
        np.std([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], axis=0),
    )


def test_global_transform_preserves_other_columns():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "label": [0, 1, 0],
        }
    )

    stats = fit_normalizer(df, ["x"])
    result = transform(df, stats)

    assert "label" in result.columns
    assert result["label"].tolist() == [0, 1, 0]
    assert np.isclose(result["x"].mean(), 0.0)


def test_regime_normalizer_fits_each_regime_separately():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 10.0, 12.0],
            "operating_regime": [0, 0, 1, 1],
        }
    )

    stats = fit_regime_normalizer(
        df,
        feature_cols=["x"],
    )

    assert stats.regimes == (0, 1)

    assert np.allclose(stats.means[0], [1.5])
    assert np.allclose(stats.means[1], [11.0])


def test_regime_transform_normalizes_each_regime():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 10.0, 12.0],
            "operating_regime": [0, 0, 1, 1],
        }
    )

    stats = fit_regime_normalizer(
        df,
        feature_cols=["x"],
    )

    result = transform_by_regime(df, stats)

    for regime in [0, 1]:
        values = result.loc[
            result["operating_regime"] == regime,
            "x",
        ].to_numpy()

        assert np.isclose(values.mean(), 0.0)
        assert np.isclose(values.std(), 1.0)


def test_regime_transform_rejects_unknown_regime():
    train = pd.DataFrame(
        {
            "x": [1.0, 2.0, 10.0, 12.0],
            "operating_regime": [0, 0, 1, 1],
        }
    )

    test = pd.DataFrame(
        {
            "x": [5.0],
            "operating_regime": [2],
        }
    )

    stats = fit_regime_normalizer(
        train,
        feature_cols=["x"],
    )

    with pytest.raises(ValueError, match="not present"):
        transform_by_regime(test, stats)