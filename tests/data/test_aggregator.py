"""Tests for BarAggregator."""

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
def sample_parquet_dir():
    """Create temporary directory with sample Parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for day in [1, 2, 3]:
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
                n_trades = 1000
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


class TestBarAggregator:
    """Tests for BarAggregator."""

    def test_aggregate_time_bars_single_symbol(self, sample_parquet_dir, column_mapping):
        """Test aggregation for a single symbol."""
        aggregator = BarAggregator()

        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 4).timestamp() * 1000)

        result, meta = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) > 0

        required_cols = ["symbol", "start_time", "end_time", "open", "high", "low", "close", "volume", "vwap"]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

        assert result["symbol"].unique().to_list() == ["BTCUSDT"]

    def test_aggregate_time_bars_multiple_symbols(self, sample_parquet_dir, column_mapping):
        """Test aggregation for multiple symbols."""
        aggregator = BarAggregator()

        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=*/**/*.parquet")

        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 4).timestamp() * 1000)

        result, meta = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT", "ETHUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        assert len(result) > 0
        assert set(result["symbol"].unique().to_list()) == {"BTCUSDT", "ETHUSDT"}

    def test_aggregate_time_bars_different_intervals(self, sample_parquet_dir, column_mapping):
        """Test that different intervals produce different bar counts."""
        aggregator = BarAggregator()

        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2).timestamp() * 1000)

        result_1m, _ = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        result_5m, _ = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=300_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        assert len(result_1m) > len(result_5m)
        assert len(result_1m) >= len(result_5m) * 4

    def test_aggregate_includes_buyer_seller_stats(self, sample_parquet_dir, column_mapping):
        """Test that buyer/seller statistics are included."""
        aggregator = BarAggregator()

        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2).timestamp() * 1000)

        result, meta = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )

        buyer_seller_cols = ["num_buyer", "num_seller", "num_buyer_volume", "num_seller_volume"]
        for col in buyer_seller_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_aggregate_without_buyer_seller_stats(self, sample_parquet_dir, column_mapping):
        """Test aggregation without buyer/seller statistics."""
        aggregator = BarAggregator()

        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2).timestamp() * 1000)

        result, meta = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
            include_buyer_seller=False,
        )

        assert "num_buyer" not in result.columns

    def test_ohlc_integrity(self, sample_parquet_dir, column_mapping):
        """Test OHLC data integrity (high >= low, etc.)."""
        aggregator = BarAggregator()

        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2).timestamp() * 1000)

        result, meta = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )

        assert (result["high"] >= result["low"]).all()
        assert (result["high"] >= result["open"]).all()
        assert (result["high"] >= result["close"]).all()
        assert (result["low"] <= result["open"]).all()
        assert (result["low"] <= result["close"]).all()
        assert (result["volume"] > 0).all()

    def test_aggregate_tick_bars_invalid_interval(self, sample_parquet_dir, column_mapping):
        """Test that invalid interval_ticks raises ValueError."""
        aggregator = BarAggregator()
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        # Test zero interval
        with pytest.raises(ValueError, match="interval_ticks must be positive"):
            aggregator.aggregate_tick_bars(
                parquet_pattern=pattern,
                symbol="BTCUSDT",
                interval_ticks=0,
                column_mapping=column_mapping,
            )

        # Test negative interval
        with pytest.raises(ValueError, match="interval_ticks must be positive"):
            aggregator.aggregate_tick_bars(
                parquet_pattern=pattern,
                symbol="BTCUSDT",
                interval_ticks=-5,
                column_mapping=column_mapping,
            )

    def test_aggregate_volume_bars_invalid_interval(self, sample_parquet_dir, column_mapping):
        """Test that invalid interval_volume raises ValueError."""
        aggregator = BarAggregator()
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        # Test zero interval
        with pytest.raises(ValueError, match="interval_volume must be positive"):
            aggregator.aggregate_volume_bars(
                parquet_pattern=pattern,
                symbol="BTCUSDT",
                interval_volume=0,
                column_mapping=column_mapping,
            )

        # Test negative interval
        with pytest.raises(ValueError, match="interval_volume must be positive"):
            aggregator.aggregate_volume_bars(
                parquet_pattern=pattern,
                symbol="BTCUSDT",
                interval_volume=-10.5,
                column_mapping=column_mapping,
            )

    def test_aggregate_dollar_bars_invalid_interval(self, sample_parquet_dir, column_mapping):
        """Test that invalid interval_dollar raises ValueError."""
        aggregator = BarAggregator()
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")

        # Test zero interval
        with pytest.raises(ValueError, match="interval_dollar must be positive"):
            aggregator.aggregate_dollar_bars(
                parquet_pattern=pattern,
                symbol="BTCUSDT",
                interval_dollar=0,
                column_mapping=column_mapping,
            )

        # Test negative interval
        with pytest.raises(ValueError, match="interval_dollar must be positive"):
            aggregator.aggregate_dollar_bars(
                parquet_pattern=pattern,
                symbol="BTCUSDT",
                interval_dollar=-100.0,
                column_mapping=column_mapping,
            )
