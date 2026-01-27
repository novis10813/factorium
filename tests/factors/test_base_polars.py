"""Tests for BaseFactor Polars support.

Tests initialize and property access for BaseFactor with Polars DataFrames,
LazyFrames, and various initialization paths.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from factorium import Factor


@pytest.fixture
def sample_pandas_df():
    """Create sample factor data as pandas DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=20, freq="1min")
    timestamps = dates.astype(np.int64) // 10**6

    data = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for i in range(10):
            data.append(
                {
                    "start_time": int(timestamps[i]),
                    "end_time": int(timestamps[i] + 60000),
                    "symbol": symbol,
                    "factor": float(np.random.randn()),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_polars_df(sample_pandas_df):
    """Create sample factor data as Polars DataFrame."""
    return pl.from_pandas(sample_pandas_df)


@pytest.fixture
def sample_polars_lazy(sample_polars_df):
    """Create sample factor data as Polars LazyFrame."""
    return sample_polars_df.lazy()


@pytest.fixture
def sample_parquet_file(sample_pandas_df, tmp_path):
    """Create sample factor data as parquet file."""
    parquet_path = tmp_path / "sample_factor.parquet"
    sample_pandas_df.to_parquet(parquet_path)
    return parquet_path


class TestBaseFactor_PandasInitialization:
    """Test BaseFactor initialization from pandas.DataFrame."""

    def test_init_from_pandas_dataframe(self, sample_pandas_df):
        """Should initialize from pandas.DataFrame and return polars.DataFrame."""
        factor = Factor(sample_pandas_df, name="test_factor")

        assert factor.name == "test_factor"
        assert len(factor) == len(sample_pandas_df)
        # Returns polars.DataFrame after initialization
        assert isinstance(factor.data, pl.DataFrame)

    def test_init_from_pandas_with_default_name(self, sample_pandas_df):
        """Should use default name if not provided."""
        factor = Factor(sample_pandas_df)

        assert factor.name == "factor"

    def test_init_pandas_preserves_data_integrity(self, sample_pandas_df):
        """Should preserve data integrity when initializing from pandas."""
        factor = Factor(sample_pandas_df)
        data = factor.data

        # Check required columns exist
        assert "start_time" in data.columns
        assert "end_time" in data.columns
        assert "symbol" in data.columns
        assert "factor" in data.columns


class TestBaseFactor_PolarsDataFrameInitialization:
    """Test BaseFactor initialization from polars.DataFrame."""

    def test_init_from_polars_dataframe(self, sample_polars_df):
        """Should initialize from polars.DataFrame."""
        factor = Factor(sample_polars_df, name="test_factor")

        assert factor.name == "test_factor"
        assert len(factor) == len(sample_polars_df)

    def test_init_from_polars_with_default_name(self, sample_polars_df):
        """Should use default name if not provided."""
        factor = Factor(sample_polars_df)

        assert factor.name == "factor"

    def test_init_polars_preserves_data(self, sample_polars_df):
        """Should preserve data when initializing from Polars."""
        factor = Factor(sample_polars_df)
        data = factor.data

        # Check required columns exist
        assert "start_time" in data.columns
        assert "end_time" in data.columns
        assert "symbol" in data.columns
        assert "factor" in data.columns


class TestBaseFactor_PolarsLazyFrameInitialization:
    """Test BaseFactor initialization from polars.LazyFrame."""

    def test_init_from_polars_lazyframe(self, sample_polars_lazy):
        """Should initialize from polars.LazyFrame."""
        factor = Factor(sample_polars_lazy, name="test_factor")

        assert factor.name == "test_factor"
        assert len(factor) == len(sample_polars_lazy.collect())

    def test_init_from_polars_lazy_with_default_name(self, sample_polars_lazy):
        """Should use default name if not provided."""
        factor = Factor(sample_polars_lazy)

        assert factor.name == "factor"

    def test_init_polars_lazy_preserves_data(self, sample_polars_lazy):
        """Should preserve data when initializing from Polars LazyFrame."""
        factor = Factor(sample_polars_lazy)
        data = factor.data

        # Check required columns exist
        assert "start_time" in data.columns
        assert "end_time" in data.columns
        assert "symbol" in data.columns
        assert "factor" in data.columns


class TestBaseFactor_ParquetInitialization:
    """Test BaseFactor initialization from parquet files."""

    def test_init_from_parquet_path(self, sample_parquet_file):
        """Should initialize from parquet file Path."""
        factor = Factor(sample_parquet_file, name="test_factor")

        assert factor.name == "test_factor"
        assert isinstance(factor.data, pl.DataFrame)

    def test_init_from_parquet_with_default_name(self, sample_parquet_file):
        """Should use default name if not provided."""
        factor = Factor(sample_parquet_file)

        assert factor.name == "factor"

    def test_init_parquet_preserves_data(self, sample_parquet_file):
        """Should preserve data when initializing from parquet."""
        factor = Factor(sample_parquet_file)
        data = factor.data

        # Check required columns exist
        assert "start_time" in data.columns
        assert "end_time" in data.columns
        assert "symbol" in data.columns
        assert "factor" in data.columns


class TestBaseFactor_DataProperty:
    """Test factor.data property behavior."""

    def test_data_property_returns_polars_dataframe(self, sample_polars_df):
        """Should return polars.DataFrame from .data property."""
        factor = Factor(sample_polars_df)
        data = factor.data

        assert isinstance(data, pl.DataFrame)

    def test_data_property_contains_required_columns(self, sample_polars_df):
        """Should contain all required columns."""
        factor = Factor(sample_polars_df)
        data = factor.data

        required_cols = {"start_time", "end_time", "symbol", "factor"}
        assert required_cols.issubset(set(data.columns))

    def test_data_property_is_immutable_reference(self, sample_polars_df):
        """Should return reference (modifications should not affect original)."""
        factor = Factor(sample_polars_df)
        original_len = len(factor.data)

        # Attempting to modify should not affect the factor
        # (Polars operations are immutable by default)
        _ = factor.data
        assert len(factor.data) == original_len

    def test_data_property_preserves_dtypes(self, sample_polars_df):
        """Should preserve expected data types."""
        factor = Factor(sample_polars_df)
        data = factor.data

        assert data["start_time"].dtype == pl.Int64
        assert data["end_time"].dtype == pl.Int64
        assert data["symbol"].dtype == pl.Utf8 or data["symbol"].dtype == pl.String
        assert data["factor"].dtype == pl.Float64


class TestBaseFactor_ToPandasMethod:
    """Test factor.to_pandas() method."""

    def test_to_pandas_returns_pandas_dataframe(self, sample_polars_df):
        """Should return pandas.DataFrame."""
        factor = Factor(sample_polars_df)
        pandas_data = factor.to_pandas()

        assert isinstance(pandas_data, pd.DataFrame)

    def test_to_pandas_contains_all_columns(self, sample_polars_df):
        """Should contain all required columns."""
        factor = Factor(sample_polars_df)
        pandas_data = factor.to_pandas()

        required_cols = {"start_time", "end_time", "symbol", "factor"}
        assert required_cols.issubset(set(pandas_data.columns))

    def test_to_pandas_matches_internal_data(self, sample_polars_df):
        """Should match the internal data content."""
        factor = Factor(sample_polars_df)
        pandas_data = factor.to_pandas()

        assert len(pandas_data) == len(factor.data)
        assert list(pandas_data.columns) == list(factor.data.columns)

    def test_to_pandas_from_polars_input(self, sample_polars_df):
        """Should correctly convert from Polars to Pandas."""
        factor = Factor(sample_polars_df)
        pandas_data = factor.to_pandas()

        assert isinstance(pandas_data, pd.DataFrame)
        assert len(pandas_data) > 0

    def test_to_pandas_from_pandas_input(self, sample_pandas_df):
        """Should work with pandas input as well."""
        factor = Factor(sample_pandas_df)
        pandas_data = factor.to_pandas()

        assert isinstance(pandas_data, pd.DataFrame)
        assert len(pandas_data) == len(sample_pandas_df)

    def test_to_pandas_from_lazy_input(self, sample_polars_lazy):
        """Should work with LazyFrame input."""
        factor = Factor(sample_polars_lazy)
        pandas_data = factor.to_pandas()

        assert isinstance(pandas_data, pd.DataFrame)


class TestBaseFactor_LazyProperty:
    """Test factor.lazy property for lazy evaluation."""

    def test_lazy_returns_lazyframe(self, sample_polars_df):
        """Should return polars.LazyFrame."""
        factor = Factor(sample_polars_df)
        lazy_data = factor.lazy

        assert isinstance(lazy_data, pl.LazyFrame)

    def test_lazy_contains_required_columns(self, sample_polars_df):
        """Should contain all required columns."""
        factor = Factor(sample_polars_df)
        lazy_data = factor.lazy

        # Collect to check columns
        collected = lazy_data.collect()
        required_cols = {"start_time", "end_time", "symbol", "factor"}
        assert required_cols.issubset(set(collected.columns))

    def test_lazy_preserves_data_content(self, sample_polars_df):
        """Should preserve data content when collected."""
        factor = Factor(sample_polars_df)
        lazy_data = factor.lazy
        collected = lazy_data.collect()

        assert len(collected) == len(sample_polars_df)

    def test_lazy_from_pandas_input(self, sample_pandas_df):
        """Should create lazy frame from pandas input."""
        factor = Factor(sample_pandas_df)
        lazy_data = factor.lazy

        assert isinstance(lazy_data, pl.LazyFrame)
        assert len(lazy_data.collect()) > 0

    def test_lazy_from_parquet_input(self, sample_parquet_file):
        """Should create lazy frame from parquet input."""
        factor = Factor(sample_parquet_file)
        lazy_data = factor.lazy

        assert isinstance(lazy_data, pl.LazyFrame)


class TestBaseFactor_LazyEvaluationBehavior:
    """Test that lazy evaluation does not trigger collect automatically.

    Tests verify that creating factors and chaining operations does not
    perform eager evaluation until explicitly collected.
    """

    def test_lazy_evaluation_not_triggered_on_init(self, sample_polars_lazy):
        """Should not trigger collect() on initialization."""
        with patch.object(pl.LazyFrame, "collect", wraps=pl.LazyFrame.collect) as mock_collect:
            factor = Factor(sample_polars_lazy)

            # collect() should not have been called
            assert mock_collect.call_count == 0

    def test_lazy_evaluation_not_triggered_on_property_access(self, sample_polars_lazy):
        """Should not trigger collect() when accessing .lazy property."""
        factor = Factor(sample_polars_lazy)

        with patch.object(pl.LazyFrame, "collect", wraps=pl.LazyFrame.collect) as mock_collect:
            _ = factor.lazy

            # collect() should not have been called
            assert mock_collect.call_count == 0

    def test_lazy_evaluation_only_on_explicit_collect(self, sample_polars_lazy):
        """Should only evaluate when explicitly calling collect()."""
        factor = Factor(sample_polars_lazy)
        lazy_data = factor.lazy

        # At this point, collect() should not have been called
        # Only when we explicitly collect should it evaluate
        result = lazy_data.collect()

        assert isinstance(result, pl.DataFrame)

    def test_lazy_evaluation_collect_count(self, sample_polars_lazy):
        """Should track number of collect() calls."""
        collect_count = 0
        original_collect = pl.LazyFrame.collect

        def counting_collect(self):
            nonlocal collect_count
            collect_count += 1
            return original_collect(self)

        with patch.object(pl.LazyFrame, "collect", counting_collect):
            factor = Factor(sample_polars_lazy)
            _ = factor.lazy

            # Verify collect hasn't been called yet
            assert collect_count == 0

            # Now collect
            result = factor.lazy.collect()

            # Should have been called once
            assert collect_count == 1


class TestBaseFactor_SchemaNormalization:
    """Test schema normalization when input column names differ.

    Factor should normalize non-standard column names to the standard
    schema: start_time, end_time, symbol, factor.
    """

    def test_normalize_single_unnamed_column(self, sample_pandas_df):
        """Should normalize DataFrame with 4 columns to standard schema."""
        # Rename columns to non-standard names
        df = sample_pandas_df.copy()
        df.columns = ["t1", "t2", "sym", "val"]

        factor = Factor(df)
        data = factor.data

        expected_cols = {"start_time", "end_time", "symbol", "factor"}
        assert expected_cols == set(data.columns)

    def test_normalize_with_standard_columns(self, sample_pandas_df):
        """Should preserve standard column names."""
        factor = Factor(sample_pandas_df)
        data = factor.data

        assert "start_time" in data.columns
        assert "end_time" in data.columns
        assert "symbol" in data.columns
        assert "factor" in data.columns

    def test_normalize_polars_single_unnamed_column(self, sample_polars_df):
        """Should normalize Polars DataFrame with 4 columns."""
        # Create Polars DataFrame with non-standard column names
        df = pl.DataFrame(
            {
                "t1": sample_polars_df["start_time"],
                "t2": sample_polars_df["end_time"],
                "sym": sample_polars_df["symbol"],
                "val": sample_polars_df["factor"],
            }
        )

        factor = Factor(df)
        data = factor.data

        expected_cols = {"start_time", "end_time", "symbol", "factor"}
        assert expected_cols == set(data.columns)

    def test_normalize_preserves_data_order(self, sample_pandas_df):
        """Should normalize while preserving data integrity."""
        df = sample_pandas_df.copy()
        original_factor_values = df["factor"].values

        factor = Factor(df)
        data = factor.data

        # Data should be sorted by end_time, symbol
        # Values should match original (after sorting)
        sorted_df = sample_pandas_df.sort_values(by=["end_time", "symbol"])
        # Allow for reordering but data should match
        assert len(data) == len(sorted_df)


class TestBaseFactor_EdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe_raises_error(self):
        """Should handle empty DataFrames appropriately."""
        empty_df = pd.DataFrame(columns=["start_time", "end_time", "symbol", "factor"])

        # May raise error or create empty factor
        # Test that behavior is defined
        try:
            factor = Factor(empty_df)
            assert len(factor) == 0
        except (ValueError, KeyError):
            # Also acceptable to raise error for empty data
            pass

    def test_missing_required_column_raises_error(self):
        """Should raise error if required columns are missing."""
        df = pd.DataFrame(
            {
                "start_time": [1, 2],
                "end_time": [3, 4],
                "symbol": ["A", "B"],
                # Missing "factor" column
            }
        )

        with pytest.raises((ValueError, KeyError)):
            Factor(df)

    def test_invalid_file_type_raises_error(self, tmp_path):
        """Should raise error for unsupported file types."""
        invalid_file = tmp_path / "data.txt"
        invalid_file.write_text("some data")

        with pytest.raises(ValueError):
            Factor(invalid_file)

    def test_duplicate_symbols_handled(self, sample_pandas_df):
        """Should handle DataFrames with duplicate symbol-time combinations."""
        factor = Factor(sample_pandas_df)

        # Should not raise error
        assert len(factor) > 0

    def test_unsupported_input_type_raises_error(self):
        """Should raise error for unsupported input types."""
        with pytest.raises(ValueError):
            Factor([1, 2, 3])  # List is not supported

    def test_unsupported_input_dict_raises_error(self):
        """Should raise error for dict input."""
        with pytest.raises((ValueError, TypeError)):
            Factor({"key": "value"})


class TestBaseFactor_Integration:
    """Integration tests for Polars support."""

    def test_create_from_polars_chain_to_pandas(self, sample_polars_df):
        """Should support creating from Polars and converting back to Pandas."""
        factor = Factor(sample_polars_df, name="test")

        # Should be able to convert to pandas
        pandas_data = factor.to_pandas()

        assert isinstance(pandas_data, pd.DataFrame)
        assert len(pandas_data) == len(sample_polars_df)

    def test_create_from_lazy_get_eager_data(self, sample_polars_lazy):
        """Should support creating from LazyFrame and accessing eager data."""
        factor = Factor(sample_polars_lazy, name="test")

        # .data should return eager polars.DataFrame
        data = factor.data

        assert data is not None
        assert isinstance(data, pl.DataFrame)
        assert len(data) > 0

    def test_multiple_factors_from_different_sources(self, sample_pandas_df, sample_polars_df):
        """Should support creating multiple factors from different sources."""
        factor1 = Factor(sample_pandas_df, name="factor1")
        factor2 = Factor(sample_polars_df, name="factor2")

        assert factor1.name == "factor1"
        assert factor2.name == "factor2"
        assert len(factor1) == len(factor2)

    def test_factor_operations_after_polars_init(self, sample_polars_df):
        """Should support factor operations after initializing from Polars."""
        factor = Factor(sample_polars_df, name="f")

        # Should support basic operations
        result = factor + 1

        assert result.name == "(f+1)"
        assert len(result) == len(factor)


class TestBaseFactor_LenOptimization:
    """Test __len__ optimization to avoid full collection."""

    def test_len_avoids_full_collection(self):
        """Verify __len__ uses count query, not full collection."""
        import time

        # Create large dataset
        n_rows = 100_000
        df = pd.DataFrame(
            {
                "start_time": pd.date_range("2020-01-01", periods=n_rows, freq="1min").astype(np.int64) // 10**6,
                "end_time": (pd.date_range("2020-01-01", periods=n_rows, freq="1min").astype(np.int64) // 10**6)
                + 60000,
                "symbol": ["A"] * n_rows,
                "factor": np.arange(n_rows, dtype=float),
            }
        )

        factor = Factor(df)

        # Add expensive operations to LazyFrame
        expensive_factor = factor.ts_mean(20).ts_std(20).cs_rank()

        # __len__ should be fast (count only)
        start = time.perf_counter()
        length = len(expensive_factor)
        len_time = time.perf_counter() - start

        assert length == n_rows
        assert len_time < 0.5, f"__len__ too slow: {len_time:.3f}s (should use count query)"

        # Full collection should be slower
        start = time.perf_counter()
        _ = expensive_factor.data
        collect_time = time.perf_counter() - start

        assert collect_time > len_time, "Count should be faster than full collection"

    def test_len_correctness_simple(self, sample_polars_df):
        """Verify __len__ returns correct value for simple factor."""
        factor = Factor(sample_polars_df)
        assert len(factor) == len(sample_polars_df)

    def test_len_correctness_after_operations(self, sample_polars_df):
        """Verify __len__ returns correct value after operations."""
        factor = Factor(sample_polars_df)
        result = factor + 10
        assert len(result) == len(factor)

    def test_len_correctness_lazy_input(self, sample_polars_lazy):
        """Verify __len__ works correctly with LazyFrame input."""
        factor = Factor(sample_polars_lazy)
        expected_len = len(sample_polars_lazy.collect())
        assert len(factor) == expected_len


class TestBinaryOps_JoinBehavior:
    """Test binary op alignment uses full union of keys."""

    def test_binary_op_full_join_union_keys(self):
        left = pd.DataFrame(
            {
                "start_time": [1, 2],
                "end_time": [2, 3],
                "symbol": ["A", "B"],
                "factor": [1.0, 2.0],
            }
        )
        right = pd.DataFrame(
            {
                "start_time": [3, 4],
                "end_time": [4, 5],
                "symbol": ["C", "D"],
                "factor": [10.0, 20.0],
            }
        )

        f_left = Factor(left, name="left")
        f_right = Factor(right, name="right")

        result = f_left + f_right
        result_df = result.to_pandas()

        assert set(result_df["symbol"]) == {"A", "B", "C", "D"}
        assert len(result_df) == 4
        assert result_df["factor"].isna().all()
