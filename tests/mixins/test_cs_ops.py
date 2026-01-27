import pytest
import pandas as pd
import numpy as np
from factorium.factors.base import BaseFactor
from factorium.factors.mixins.cs_ops import CrossSectionalOpsMixin


class Factor(BaseFactor, CrossSectionalOpsMixin):
    pass


@pytest.fixture
def sample_data():
    dates = pd.to_datetime(["2023-01-01", "2023-01-02"])
    symbols = ["A", "B", "C"]
    data = []
    for d in dates:
        for i, s in enumerate(symbols):
            data.append(
                {
                    "start_time": d,
                    "end_time": d,
                    "symbol": s,
                    "factor": float(i + 1),  # 1.0, 2.0, 3.0
                }
            )
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_nan():
    dates = pd.to_datetime(["2023-01-01"])
    symbols = ["A", "B", "C"]
    data = [
        {"start_time": dates[0], "end_time": dates[0], "symbol": "A", "factor": 1.0},
        {"start_time": dates[0], "end_time": dates[0], "symbol": "B", "factor": np.nan},
        {"start_time": dates[0], "end_time": dates[0], "symbol": "C", "factor": 3.0},
    ]
    return pd.DataFrame(data)


def test_cs_rank(sample_data):
    f = Factor(sample_data)
    ranked = f.cs_rank()

    # Check for first date
    df = ranked.to_pandas()
    date1 = pd.to_datetime("2023-01-01")
    group1 = df[df["end_time"] == date1]

    # Values are 1, 2, 3 -> Ranks should be 1/3, 2/3, 3/3 approx
    expected = np.array([1 / 3, 2 / 3, 3 / 3])  # method='min' gives 0.33, 0.66, 1.0
    np.testing.assert_allclose(group1["factor"].to_numpy(), expected, rtol=1e-5)


def test_cs_rank_strict_nan(sample_data_with_nan):
    f = Factor(sample_data_with_nan)
    ranked = f.cs_rank()

    # Should be all NaN because one value is NaN
    assert pd.isna(ranked.to_pandas()["factor"]).all()


def test_cs_zscore(sample_data):
    f = Factor(sample_data)
    zscored = f.cs_zscore()

    df = zscored.to_pandas()
    date1 = pd.to_datetime("2023-01-01")
    group1 = df[df["end_time"] == date1]

    # Values: 1, 2, 3. Mean=2, Std=1. Z = [-1, 0, 1]
    expected = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(group1["factor"].to_numpy(), expected, atol=1e-5)


def test_cs_zscore_strict_nan(sample_data_with_nan):
    f = Factor(sample_data_with_nan)
    zscored = f.cs_zscore()
    assert pd.isna(zscored.to_pandas()["factor"]).all()


def test_cs_demean(sample_data):
    f = Factor(sample_data)
    demeaned = f.cs_demean()

    df = demeaned.to_pandas()
    date1 = pd.to_datetime("2023-01-01")
    group1 = df[df["end_time"] == date1]

    # Values: 1, 2, 3. Mean=2. Demeaned = [-1, 0, 1]
    expected = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(group1["factor"].to_numpy(), expected, atol=1e-5)


def test_cs_winsorize(sample_data):
    # Create data with outliers: 1, 2, 3, 100
    dates = pd.to_datetime(["2023-01-01"])
    data = pd.DataFrame(
        [
            {"start_time": dates[0], "end_time": dates[0], "symbol": "A", "factor": 1.0},
            {"start_time": dates[0], "end_time": dates[0], "symbol": "B", "factor": 2.0},
            {"start_time": dates[0], "end_time": dates[0], "symbol": "C", "factor": 3.0},
            {"start_time": dates[0], "end_time": dates[0], "symbol": "D", "factor": 100.0},
        ]
    )
    f = Factor(data)

    # Winsorize at 25% (will clip to 25th and 75th percentile)
    # 25th percentile of [1, 2, 3, 100] is roughly 1.75
    # 75th percentile is roughly 27.25
    # So 1 -> 1.75, 100 -> 27.25
    winsorized = f.cs_winsorize(0.25)
    vals = winsorized.to_pandas()["factor"].to_numpy()

    # Check that extreme values are clipped (exact values depend on interpolation)
    assert vals[3] < 100.0
    assert vals[0] >= 1.0


def test_cs_neutralize():
    dates = pd.to_datetime(["2023-01-01"])
    # Y = 2*X + 1
    data_y = pd.DataFrame(
        [
            {"start_time": dates[0], "end_time": dates[0], "symbol": "A", "factor": 3.0},  # X=1
            {"start_time": dates[0], "end_time": dates[0], "symbol": "B", "factor": 5.0},  # X=2
            {"start_time": dates[0], "end_time": dates[0], "symbol": "C", "factor": 7.0},  # X=3
        ]
    )
    data_x = pd.DataFrame(
        [
            {"start_time": dates[0], "end_time": dates[0], "symbol": "A", "factor": 1.0},
            {"start_time": dates[0], "end_time": dates[0], "symbol": "B", "factor": 2.0},
            {"start_time": dates[0], "end_time": dates[0], "symbol": "C", "factor": 3.0},
        ]
    )

    f_y = Factor(data_y, name="Y")
    f_x = Factor(data_x, name="X")

    neutralized = f_y.cs_neutralize(f_x)

    # Should be close to 0 residuals (perfect linear relationship)
    np.testing.assert_allclose(neutralized.to_pandas()["factor"].to_numpy(), 0.0, atol=1e-10)


def test_cs_neutralize_nan(sample_data_with_nan, sample_data):
    # If Y has NaN, result should be NaN
    f_y = Factor(sample_data_with_nan)
    # X has valid data (take only first date to match)
    f_x = Factor(sample_data.iloc[:3].copy())

    neutralized = f_y.cs_neutralize(f_x)
    assert pd.isna(neutralized.to_pandas()["factor"]).all()
