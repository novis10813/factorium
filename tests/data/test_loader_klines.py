"""Tests for BinanceDataLoader.load_aggbar with klines data."""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import patch, MagicMock

from factorium.data.loader import BinanceDataLoader
from factorium.aggbar import AggBar


@pytest.fixture
def sample_klines_data():
    """Create temporary directory with Hive-partitioned klines Parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for day in [1, 2, 3]:
                partition_path = (
                    tmpdir
                    / "market=futures_um"
                    / "data_type=klines"
                    / f"symbol={symbol}"
                    / "year=2024"
                    / "month=01"
                    / f"day={day:02d}"
                )
                partition_path.mkdir(parents=True, exist_ok=True)

                # Klines data: 1440 bars per day (1 minute bars)
                n_bars = 1440
                base_ts = int(datetime(2024, 1, day).timestamp() * 1000)

                df = pd.DataFrame(
                    {
                        "open_time": base_ts + np.arange(n_bars) * 60000,
                        "open": 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1),
                        "high": 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1) + 0.5,
                        "low": 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1) - 0.5,
                        "close": 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1),
                        "volume": np.abs(np.random.randn(n_bars)) * 100 + 10,
                        "close_time": base_ts + np.arange(n_bars) * 60000 + 59999,
                        "quote_volume": np.abs(np.random.randn(n_bars)) * 1000,
                        "count": np.random.randint(10, 100, n_bars),
                        "taker_buy_volume": np.abs(np.random.randn(n_bars)) * 50,
                        "taker_buy_quote_volume": np.abs(np.random.randn(n_bars)) * 500,
                        "ignore": np.zeros(n_bars),
                    }
                )

                table = pa.Table.from_pandas(df)
                pq.write_table(table, partition_path / "data.parquet")

        yield tmpdir


class TestLoadKlines:
    """Tests for BinanceDataLoader.load_aggbar with klines."""

    def test_returns_aggbar_for_klines(self, sample_klines_data):
        """Test that load_aggbar returns AggBar instance for klines."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=3,
                use_cache=False,
            )

        assert isinstance(result, AggBar)
        assert "BTCUSDT" in result.symbols

    def test_klines_has_all_columns(self, sample_klines_data):
        """Test that klines data has all expected columns."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                use_cache=False,
            )

        # All klines columns including microstructure data
        required_cols = {
            "symbol",
            "start_time",
            "end_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        }
        assert required_cols.issubset(set(result.cols))

    def test_klines_bypasses_aggregation(self, sample_klines_data):
        """Test that klines loading bypasses BarAggregator."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch("factorium.data.loader.BarAggregator") as MockAggregator:
            mock_agg_instance = MockAggregator.return_value
            mock_agg_instance.aggregate_time_bars.return_value = (
                pl.DataFrame({"symbol": ["BTCUSDT"], "start_time": [1], "end_time": [2]}),
                MagicMock(),
            )

            with patch.object(loader, "_check_all_symbols_exist", return_value=True):
                result = loader.load_aggbar(
                    symbols=["BTCUSDT"],
                    data_type="klines",
                    market_type="futures",
                    futures_type="um",
                    start_date="2024-01-01",
                    days=1,
                    use_cache=False,
                )

            # Aggregator should not be called for klines
            assert not mock_agg_instance.aggregate_time_bars.called
            assert not mock_agg_instance.aggregate_tick_bars.called
            assert not mock_agg_instance.aggregate_volume_bars.called
            assert not mock_agg_instance.aggregate_dollar_bars.called

    def test_klines_loads_multiple_symbols(self, sample_klines_data):
        """Test loading klines for multiple symbols."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result = loader.load_aggbar(
                symbols=["BTCUSDT", "ETHUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=3,
                use_cache=False,
            )

        assert isinstance(result, AggBar)
        assert set(result.symbols) == {"BTCUSDT", "ETHUSDT"}

    def test_klines_resample_to_5m(self, sample_klines_data):
        """Test resampling 1m klines to 5m."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result_1m = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                interval=60_000,  # 1m
                use_cache=False,
            )

            result_5m = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                interval=300_000,  # 5m
                use_cache=False,
            )

        # 5m should have 1/5 the bars of 1m
        assert len(result_5m) == len(result_1m) // 5

    def test_klines_resample_to_1h(self, sample_klines_data):
        """Test resampling 1m klines to 1h."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result_1m = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                interval=60_000,  # 1m
                use_cache=False,
            )

            result_1h = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                interval=3_600_000,  # 1h
                use_cache=False,
            )

        # 1h should have 1/60 the bars of 1m
        assert len(result_1h) == len(result_1m) // 60

    def test_klines_raises_on_non_time_bar_type(self, sample_klines_data):
        """Test that klines raises error for non-time bar types."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            with pytest.raises(ValueError, match="only supports bar_type='time'"):
                loader.load_aggbar(
                    symbols=["BTCUSDT"],
                    data_type="klines",
                    market_type="futures",
                    futures_type="um",
                    start_date="2024-01-01",
                    days=1,
                    bar_type="tick",  # Should fail
                    interval=1000,
                    use_cache=False,
                )
