"""Tests for PolarsEngine."""

import pytest
import polars as pl
import pandas as pd
import numpy as np

from factorium.factors.engine import PolarsEngine


@pytest.fixture
def sample_factor_df():
    """Create sample factor data as Polars DataFrame."""
    np.random.seed(42)
    n_rows = 100

    dates = pd.date_range("2024-01-01", periods=50, freq="1min")
    timestamps = dates.astype(np.int64) // 10**6

    return pl.DataFrame(
        {
            "start_time": pl.Series(list(timestamps) * 2),
            "end_time": pl.Series(list(timestamps + 60000) * 2),
            "symbol": ["BTCUSDT"] * 50 + ["ETHUSDT"] * 50,
            "factor": pl.Series(np.random.randn(n_rows)),
        }
    )


class TestPolarsEngine:
    """Tests for PolarsEngine."""

    @pytest.fixture
    def engine(self):
        return PolarsEngine()

    def test_ts_mean(self, sample_factor_df, engine):
        """Test rolling mean."""
        result = engine.ts_mean(sample_factor_df, window=5)

        assert isinstance(result, pl.DataFrame)
        assert "factor" in result.columns
        assert len(result) == len(sample_factor_df)

        # First 4 values per symbol should be null (window not full)
        btc_data = result.filter(pl.col("symbol") == "BTCUSDT")
        assert btc_data["factor"][:4].null_count() == 4

    def test_ts_std(self, sample_factor_df, engine):
        """Test rolling standard deviation."""
        result = engine.ts_std(sample_factor_df, window=5)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_factor_df)

    def test_ts_shift(self, sample_factor_df, engine):
        """Test shift operation."""
        result = engine.ts_shift(sample_factor_df, period=3)

        assert isinstance(result, pl.DataFrame)

        # First 3 values per symbol should be null
        btc_data = result.filter(pl.col("symbol") == "BTCUSDT")
        assert btc_data["factor"][:3].null_count() == 3

    def test_ts_diff(self, sample_factor_df, engine):
        """Test diff operation."""
        result = engine.ts_diff(sample_factor_df, period=1)

        assert isinstance(result, pl.DataFrame)

        # First value per symbol should be null
        btc_data = result.filter(pl.col("symbol") == "BTCUSDT")
        assert btc_data["factor"][0] is None

    def test_cs_rank(self, sample_factor_df, engine):
        """Test cross-sectional rank."""
        result = engine.cs_rank(sample_factor_df)

        assert isinstance(result, pl.DataFrame)

        # Ranks should be between 0 and 1
        assert result["factor"].min() >= 0
        assert result["factor"].max() <= 1

    def test_cs_zscore(self, sample_factor_df, engine):
        """Test cross-sectional z-score."""
        result = engine.cs_zscore(sample_factor_df)

        assert isinstance(result, pl.DataFrame)

        # Z-scores for each time should have mean ~0
        for end_time in result["end_time"].unique():
            time_data = result.filter(pl.col("end_time") == end_time)
            mean = time_data["factor"].mean()
            assert abs(mean) < 0.1  # Close to 0

    def test_to_pandas(self, sample_factor_df, engine):
        """Test conversion to Pandas."""
        result = engine.to_pandas(sample_factor_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_factor_df)

    def test_from_pandas(self, engine):
        """Test conversion from Pandas."""
        pdf = pd.DataFrame(
            {
                "start_time": [1, 2, 3],
                "end_time": [2, 3, 4],
                "symbol": ["A", "A", "A"],
                "factor": [1.0, 2.0, 3.0],
            }
        )

        result = engine.from_pandas(pdf)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(pdf)


class TestStrictNaNHandling:
    """Tests for strict NaN propagation behavior."""

    @pytest.fixture
    def engine(self):
        return PolarsEngine()

    def test_ts_mean_with_nan_in_window(self, engine):
        """Test that NaN in rolling window produces NaN output."""
        df = pl.DataFrame(
            {
                "symbol": ["A"] * 5,
                "end_time": [1, 2, 3, 4, 5],
                "factor": [1.0, None, 3.0, 4.0, 5.0],
            }
        )

        result = engine.ts_mean(df, window=3)
        factors = result["factor"].to_list()

        assert factors[2] is None
        assert factors[3] is None

    def test_ts_std_with_nan_in_window(self, engine):
        """Test that NaN in rolling std window produces NaN output."""
        df = pl.DataFrame(
            {
                "symbol": ["A"] * 5,
                "end_time": [1, 2, 3, 4, 5],
                "factor": [1.0, None, 3.0, 4.0, 5.0],
            }
        )

        result = engine.ts_std(df, window=3)
        factors = result["factor"].to_list()

        assert factors[2] is None
        assert factors[3] is None

    def test_cs_rank_with_nan_propagates(self, engine):
        """Test that NaN in cross-section propagates to all symbols at that time."""
        df = pl.DataFrame(
            {
                "symbol": ["A", "B", "C", "A", "B", "C"],
                "end_time": [1, 1, 1, 2, 2, 2],
                "factor": [1.0, None, 3.0, 4.0, 5.0, 6.0],
            }
        )

        result = engine.cs_rank(df)
        time1 = result.filter(pl.col("end_time") == 1)["factor"].to_list()
        time2 = result.filter(pl.col("end_time") == 2)["factor"].to_list()

        assert all(v is None for v in time1)
        assert all(v is not None for v in time2)

    def test_cs_zscore_with_nan_propagates(self, engine):
        """Test that NaN in cross-section z-score propagates."""
        df = pl.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "end_time": [1, 1, 1],
                "factor": [1.0, None, 3.0],
            }
        )

        result = engine.cs_zscore(df)
        factors = result["factor"].to_list()

        assert all(v is None for v in factors)

    def test_cs_demean_with_nan_propagates(self, engine):
        """Test that NaN in cross-section demean propagates."""
        df = pl.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "end_time": [1, 1, 1],
                "factor": [1.0, None, 3.0],
            }
        )

        result = engine.cs_demean(df)
        factors = result["factor"].to_list()

        assert all(v is None for v in factors)
