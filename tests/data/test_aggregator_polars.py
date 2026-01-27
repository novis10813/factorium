"""Tests for BarAggregator Polars migration.

This test suite validates that BarAggregator returns Polars DataFrames
with AggBarMetadata instead of Pandas DataFrames.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

from factorium.data.aggregator import BarAggregator
from factorium.data.adapters import ColumnMapping
from factorium.data.metadata import AggBarMetadata


@pytest.fixture
def temp_parquet_dir():
    """Create temporary directory with sample Parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for day in [1, 2]:
                partition_path = (
                    tmpdir
                    / "market=futures_um"
                    / "data_type=aggTrades"
                    / f"symbol={symbol}"
                    / "year=2024"
                    / "month=01"
                    / f"day={day:02d}"
                )
                partition_path.mkdir(parents=True, exist_ok=True)

                np.random.seed(42 + day + hash(symbol) % 100)
                n_trades = 500
                base_ts = int(datetime(2024, 1, day).timestamp() * 1000)

                df = pd.DataFrame(
                    {
                        "transact_time": base_ts + np.arange(n_trades) * 60000,
                        "price": 100.0 + np.cumsum(np.random.randn(n_trades) * 0.1),
                        "quantity": np.abs(np.random.randn(n_trades)) * 10 + 1,
                        "is_buyer_maker": np.random.choice([True, False], n_trades),
                    }
                )

                table = pa.Table.from_pandas(df)
                pq.write_table(table, partition_path / "data.parquet")

        yield tmpdir


@pytest.fixture
def column_mapping():
    return ColumnMapping(
        timestamp="transact_time",
        price="price",
        volume="quantity",
        is_buyer_maker="is_buyer_maker",
    )


class TestBarAggregatorPolars:
    """Tests for Polars DataFrame + metadata return type."""

    def test_aggregate_time_bars_returns_tuple(self, temp_parquet_dir, column_mapping):
        """Test that aggregate_time_bars returns (pl.DataFrame, AggBarMetadata) tuple."""
        aggregator = BarAggregator()
        pattern = str(temp_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 3).timestamp() * 1000)

        result = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        # Should return a tuple
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"

        df, metadata = result

        # DataFrame should be Polars
        assert isinstance(df, pl.DataFrame), f"Expected pl.DataFrame, got {type(df)}"

        # Metadata should be AggBarMetadata
        assert isinstance(metadata, AggBarMetadata), f"Expected AggBarMetadata, got {type(metadata)}"

    def test_aggregate_time_bars_has_correct_schema(self, temp_parquet_dir, column_mapping):
        """Test that Polars DataFrame has expected columns."""
        aggregator = BarAggregator()
        pattern = str(temp_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 3).timestamp() * 1000)

        df, metadata = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        required_cols = ["symbol", "start_time", "end_time", "open", "high", "low", "close", "volume", "vwap"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Should have buyer/seller stats by default
        optional_cols = ["num_buyer", "num_seller", "num_buyer_volume", "num_seller_volume"]
        for col in optional_cols:
            assert col in df.columns, f"Missing optional column: {col}"

    def test_metadata_reflects_actual_data(self, temp_parquet_dir, column_mapping):
        """Test that metadata is computed from actual data, not parameters."""
        aggregator = BarAggregator()
        pattern = str(temp_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 3).timestamp() * 1000)

        df, metadata = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        # Metadata should have actual data properties
        assert metadata.symbols == ["BTCUSDT"], f"Expected ['BTCUSDT'], got {metadata.symbols}"
        assert metadata.num_rows == len(df), f"Metadata rows {metadata.num_rows} != df rows {len(df)}"
        assert metadata.min_time == df["start_time"].min(), "Metadata min_time doesn't match actual data"
        assert metadata.max_time == df["end_time"].max(), "Metadata max_time doesn't match actual data"

    def test_aggregate_tick_bars_returns_polars_and_metadata(self, temp_parquet_dir, column_mapping):
        """Test that aggregate_tick_bars returns (pl.DataFrame, AggBarMetadata)."""
        aggregator = BarAggregator()
        pattern = str(temp_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        df, metadata = aggregator.aggregate_tick_bars(
            parquet_pattern=pattern,
            symbol="BTCUSDT",
            interval_ticks=50,
            column_mapping=column_mapping,
        )

        assert isinstance(df, pl.DataFrame)
        assert isinstance(metadata, AggBarMetadata)
        assert metadata.num_rows == len(df)

    def test_aggregate_volume_bars_returns_polars_and_metadata(self, temp_parquet_dir, column_mapping):
        """Test that aggregate_volume_bars returns (pl.DataFrame, AggBarMetadata)."""
        aggregator = BarAggregator()
        pattern = str(temp_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        df, metadata = aggregator.aggregate_volume_bars(
            parquet_pattern=pattern,
            symbol="BTCUSDT",
            interval_volume=100.0,
            column_mapping=column_mapping,
        )

        assert isinstance(df, pl.DataFrame)
        assert isinstance(metadata, AggBarMetadata)
        assert metadata.num_rows == len(df)

    def test_aggregate_dollar_bars_returns_polars_and_metadata(self, temp_parquet_dir, column_mapping):
        """Test that aggregate_dollar_bars returns (pl.DataFrame, AggBarMetadata)."""
        aggregator = BarAggregator()
        pattern = str(temp_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        df, metadata = aggregator.aggregate_dollar_bars(
            parquet_pattern=pattern,
            symbol="BTCUSDT",
            interval_dollar=10000.0,
            column_mapping=column_mapping,
        )

        assert isinstance(df, pl.DataFrame)
        assert isinstance(metadata, AggBarMetadata)
        assert metadata.num_rows == len(df)

    def test_empty_result_returns_empty_dataframe_and_metadata(self, temp_parquet_dir, column_mapping):
        """Test that empty results return empty DataFrame and zeroed metadata."""
        aggregator = BarAggregator()
        pattern = str(temp_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        # Request data outside the time range
        start_ts = int(datetime(2030, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2030, 1, 2).timestamp() * 1000)

        df, metadata = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        assert df.is_empty(), "Expected empty DataFrame"
        assert metadata.symbols == [], "Expected empty symbols list"
        assert metadata.num_rows == 0, "Expected 0 rows in metadata"
        assert metadata.min_time == 0, "Expected min_time=0 for empty data"
        assert metadata.max_time == 0, "Expected max_time=0 for empty data"

    def test_multiple_symbols_metadata(self, temp_parquet_dir, column_mapping):
        """Test metadata with multiple symbols."""
        aggregator = BarAggregator()
        pattern = str(temp_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=*/**/*.parquet")
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 3).timestamp() * 1000)

        df, metadata = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT", "ETHUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        # Metadata should include both symbols in sorted order
        assert set(metadata.symbols) == {"BTCUSDT", "ETHUSDT"}
        assert metadata.num_rows == len(df)
        assert metadata.min_time <= metadata.max_time
