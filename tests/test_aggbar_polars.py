"""Tests for AggBar migrated to use Polars DataFrame internally."""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime

from factorium import AggBar
from factorium.data.metadata import AggBarMetadata
from factorium.factors.core import Factor


@pytest.fixture
def sample_polars_df():
    """Create a sample Polars DataFrame for testing."""
    return pl.DataFrame(
        {
            "start_time": [1000, 2000, 3000, 1000, 2000, 3000],
            "end_time": [1100, 2100, 3100, 1100, 2100, 3100],
            "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "open": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
            "high": [102.0, 103.0, 104.0, 202.0, 203.0, 204.0],
            "low": [99.0, 100.0, 101.0, 199.0, 200.0, 201.0],
            "close": [101.0, 102.0, 103.0, 201.0, 202.0, 203.0],
            "volume": [1000, 2000, 3000, 4000, 5000, 6000],
        }
    )


@pytest.fixture
def sample_pandas_df(sample_polars_df):
    """Create a sample Pandas DataFrame for testing."""
    return sample_polars_df.to_pandas()


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return AggBarMetadata(
        symbols=["AAPL", "MSFT"],
        min_time=1000,
        max_time=3100,
        num_rows=6,
    )


class TestAggBarPolarsBasics:
    """Test basic AggBar functionality with Polars backend."""

    @pytest.fixture
    def sample_pandas_df(self, sample_polars_df):
        """Create a sample Pandas DataFrame for testing."""
        return sample_polars_df.to_pandas()

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return AggBarMetadata(
            symbols=["AAPL", "MSFT"],
            min_time=1000,
            max_time=3100,
            num_rows=6,
        )

    def test_create_from_polars_dataframe(self, sample_polars_df, sample_metadata):
        """Test creating AggBar from Polars DataFrame with metadata."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        assert agg is not None
        assert len(agg) == 6
        assert agg.metadata == sample_metadata

    def test_create_from_polars_dataframe_without_metadata(self, sample_polars_df):
        """Test creating AggBar from Polars DataFrame without metadata (compute it)."""
        agg = AggBar(sample_polars_df)
        assert agg is not None
        assert len(agg) == 6
        assert agg.metadata.symbols == ["AAPL", "MSFT"]
        assert agg.metadata.min_time == 1000
        assert agg.metadata.max_time == 3100
        assert agg.metadata.num_rows == 6

    def test_backward_compat_from_pandas(self, sample_pandas_df, sample_metadata):
        """Test backward compatibility with Pandas DataFrame input."""
        agg = AggBar(sample_pandas_df, metadata=sample_metadata)
        assert agg is not None
        assert len(agg) == 6
        # Internal storage should be Polars
        assert isinstance(agg._data, pl.DataFrame)

    def test_symbols_from_metadata(self, sample_polars_df, sample_metadata):
        """Test that symbols come from metadata (not computed from data)."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        assert agg.symbols == ["AAPL", "MSFT"]

    def test_to_df_returns_pandas(self, sample_polars_df, sample_metadata):
        """Test to_df() returns a Pandas DataFrame."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        df = agg.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert list(df.columns) == list(sample_polars_df.columns)

    def test_to_polars_returns_copy(self, sample_polars_df, sample_metadata):
        """Test to_polars() returns a copy of the Polars DataFrame."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        df = agg.to_polars()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 6
        # Ensure it's a copy
        assert df is not agg._data

    def test_cols_property(self, sample_polars_df, sample_metadata):
        """Test cols property returns column names."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        cols = agg.cols
        assert isinstance(cols, list)
        assert "start_time" in cols
        assert "end_time" in cols
        assert "symbol" in cols
        assert "close" in cols

    def test_getitem_returns_factor(self, sample_polars_df, sample_metadata):
        """Test __getitem__ with single column returns Factor."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        close_factor = agg["close"]
        assert isinstance(close_factor, Factor)
        assert close_factor.name == "close"

    def test_getitem_returns_aggbar_for_list(self, sample_polars_df, sample_metadata):
        """Test __getitem__ with list of columns returns new AggBar."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        sub_agg = agg[["close", "open"]]
        assert isinstance(sub_agg, AggBar)
        assert len(sub_agg) == 6
        assert "close" in sub_agg.cols
        assert "open" in sub_agg.cols

    def test_slice_returns_new_aggbar(self, sample_polars_df, sample_metadata):
        """Test slice() returns a new AggBar with filtered data."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        # Use 13-digit millisecond timestamps that match our data
        sliced = agg.slice(start=1000, end=2500)
        assert isinstance(sliced, AggBar)
        assert len(sliced) == 4  # Should contain AAPL(1000-1100) and MSFT(1000-1100)

    def test_slice_by_symbols(self, sample_polars_df, sample_metadata):
        """Test slice() filters by symbols."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        sliced = agg.slice(symbols=["AAPL"])
        assert len(sliced) == 3  # Only 3 AAPL rows

    def test_info_returns_pandas_summary(self, sample_polars_df, sample_metadata):
        """Test info() returns a Pandas DataFrame summary."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        info = agg.info()
        assert isinstance(info, pd.DataFrame)
        assert len(info) == 2  # One row per symbol
        assert "num_kbar" in info.columns
        assert "start_time" in info.columns
        assert "end_time" in info.columns

    def test_internal_storage_is_polars(self, sample_polars_df, sample_metadata):
        """Test that internal storage _data is Polars DataFrame."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        assert isinstance(agg._data, pl.DataFrame)

    def test_metadata_property(self, sample_polars_df, sample_metadata):
        """Test metadata property returns the metadata."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        assert agg.metadata == sample_metadata


class TestAggBarDataValidation:
    """Test data validation and error handling."""

    def test_missing_start_time_column(self):
        """Test that missing start_time column raises ValueError."""
        df = pl.DataFrame(
            {
                "end_time": [1100, 2100],
                "symbol": ["AAPL", "MSFT"],
            }
        )
        with pytest.raises(ValueError, match="must contain columns"):
            AggBar(df)

    def test_missing_end_time_column(self):
        """Test that missing end_time column raises ValueError."""
        df = pl.DataFrame(
            {
                "start_time": [1000, 2000],
                "symbol": ["AAPL", "MSFT"],
            }
        )
        with pytest.raises(ValueError, match="must contain columns"):
            AggBar(df)

    def test_missing_symbol_column(self):
        """Test that missing symbol column raises ValueError."""
        df = pl.DataFrame(
            {
                "start_time": [1000, 2000],
                "end_time": [1100, 2100],
            }
        )
        with pytest.raises(ValueError, match="must contain columns"):
            AggBar(df)

    def test_invalid_data_type(self):
        """Test that invalid data type raises TypeError."""
        with pytest.raises(TypeError, match="Invalid data type"):
            AggBar("invalid")

    def test_getitem_missing_column(self, sample_polars_df, sample_metadata):
        """Test __getitem__ with missing column raises KeyError."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        with pytest.raises(KeyError, match="Column nonexistent not found"):
            _ = agg["nonexistent"]


class TestAggBarSorting:
    """Test that data is properly sorted."""

    def test_data_sorted_by_end_time_and_symbol(self):
        """Test that data is sorted by end_time and symbol."""
        df = pl.DataFrame(
            {
                "start_time": [3000, 1000, 2000],
                "end_time": [3100, 1100, 2100],
                "symbol": ["MSFT", "AAPL", "MSFT"],
                "close": [100.0, 200.0, 300.0],
            }
        )
        agg = AggBar(df)
        collected = agg._data.to_pandas()
        # Should be sorted by end_time first, then symbol
        assert list(collected["end_time"]) == [1100, 2100, 3100]
        assert list(collected["symbol"]) == ["AAPL", "MSFT", "MSFT"]

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pl.DataFrame(
            {
                "start_time": pl.Series([], dtype=pl.Int64),
                "end_time": pl.Series([], dtype=pl.Int64),
                "symbol": pl.Series([], dtype=pl.Utf8),
            }
        )
        agg = AggBar(df)
        assert len(agg) == 0
        assert agg.metadata.symbols == []
        assert agg.metadata.num_rows == 0


class TestAggBarTimestamps:
    """Test timestamp handling."""

    def test_timestamps_property(self):
        """Test timestamps property returns unique timestamps."""
        df = pl.DataFrame(
            {
                "start_time": [1000, 2000, 1000],
                "end_time": [1100, 2100, 1100],
                "symbol": ["AAPL", "MSFT", "MSFT"],
            }
        )
        agg = AggBar(df)
        ts = agg.timestamps
        assert isinstance(ts, pd.DatetimeIndex)
        # Should have 4 unique timestamps: 1000, 1100, 2000, 2100 (in ms)
        assert len(ts) > 0


class TestAggBarCopy:
    """Test copy functionality."""

    def test_copy_creates_new_instance(self, sample_polars_df, sample_metadata):
        """Test copy() creates a new independent instance."""
        agg1 = AggBar(sample_polars_df, metadata=sample_metadata)
        agg2 = agg1.copy()
        assert agg2 is not agg1
        assert isinstance(agg2, AggBar)
        assert len(agg2) == len(agg1)


class TestAggBarLen:
    """Test __len__ method."""

    def test_len(self, sample_polars_df, sample_metadata):
        """Test __len__ returns correct row count."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        assert len(agg) == 6


class TestAggBarRepr:
    """Test __repr__ method."""

    def test_repr_contains_key_info(self, sample_polars_df, sample_metadata):
        """Test __repr__ contains key information."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        repr_str = repr(agg)
        assert "AggBar" in repr_str
        assert "6 rows" in repr_str
        assert "symbols=2" in repr_str


class TestAggBarFileIO:
    """Test file I/O methods."""

    def test_to_csv(self, sample_polars_df, sample_metadata, tmp_path):
        """Test to_csv() saves to file."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        csv_path = tmp_path / "test.csv"
        result = agg.to_csv(csv_path)
        assert result == csv_path
        assert csv_path.exists()

    def test_to_parquet(self, sample_polars_df, sample_metadata, tmp_path):
        """Test to_parquet() saves to file."""
        agg = AggBar(sample_polars_df, metadata=sample_metadata)
        parquet_path = tmp_path / "test.parquet"
        result = agg.to_parquet(parquet_path)
        assert result == parquet_path
        assert parquet_path.exists()

    def test_from_csv(self, sample_polars_df, tmp_path):
        """Test from_csv() loads from file."""
        # Create a CSV file first
        csv_path = tmp_path / "test.csv"
        sample_polars_df.write_csv(csv_path)

        # Load using from_csv
        agg = AggBar.from_csv(csv_path)
        assert agg is not None
        assert len(agg) == 6

    def test_from_df(self, sample_polars_df):
        """Test from_df() class method."""
        agg = AggBar.from_df(sample_polars_df.to_pandas())
        assert agg is not None
        assert len(agg) == 6
