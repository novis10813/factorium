"""Tests for BinanceDataLoader.load_aggbar method with time bars."""

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
def sample_hive_data():
    """Create temporary directory with Hive-partitioned Parquet files."""
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


class TestLoadAggbarTimeBars:
    """Tests for BinanceDataLoader.load_aggbar with time bars."""

    def test_returns_aggbar(self, sample_hive_data):
        """Test that load_aggbar returns AggBar instance."""
        loader = BinanceDataLoader(base_path=sample_hive_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=3,
                bar_type="time",
                interval=60_000,
                use_cache=False,
            )

        assert isinstance(result, AggBar)
        assert "BTCUSDT" in result.symbols

    def test_loads_multiple_symbols(self, sample_hive_data):
        """Test loading multiple symbols."""
        loader = BinanceDataLoader(base_path=sample_hive_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result = loader.load_aggbar(
                symbols=["BTCUSDT", "ETHUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=3,
                bar_type="time",
                interval=60_000,
                use_cache=False,
            )

        assert isinstance(result, AggBar)
        assert set(result.symbols) == {"BTCUSDT", "ETHUSDT"}

    def test_uses_cache_when_enabled(self, sample_hive_data, tmp_path):
        """Test that cache is used when enabled."""
        loader = BinanceDataLoader(base_path=sample_hive_data)

        with patch("factorium.data.loader.BarCache") as MockCache:
            mock_cache_instance = MagicMock()
            mock_cache_instance.get.return_value = None
            MockCache.return_value = mock_cache_instance

            with patch.object(loader, "_check_all_symbols_exist", return_value=True):
                loader.load_aggbar(
                    symbols=["BTCUSDT"],
                    data_type="aggTrades",
                    market_type="futures",
                    futures_type="um",
                    start_date="2024-01-01",
                    days=1,
                    bar_type="time",
                    interval=60_000,
                    use_cache=True,
                )

            assert mock_cache_instance.get.called
            assert mock_cache_instance.put.called

    def test_cache_hit_skips_aggregation(self, sample_hive_data):
        """Test that cache hit skips DuckDB aggregation."""
        loader = BinanceDataLoader(base_path=sample_hive_data)

        cached_df = pl.DataFrame(
            {
                "symbol": ["BTCUSDT"],
                "start_time": [1704067200000],
                "end_time": [1704067260000],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000.0],
                "vwap": [100.25],
            }
        )

        with patch("factorium.data.loader.BarCache") as MockCache:
            mock_cache_instance = MagicMock()
            mock_cache_instance.get.return_value = cached_df
            MockCache.return_value = mock_cache_instance

            with patch("factorium.data.loader.BarAggregator") as MockAggregator:
                mock_agg_instance = MagicMock()
                MockAggregator.return_value = mock_agg_instance

                with patch.object(loader, "_check_all_symbols_exist", return_value=True):
                    result = loader.load_aggbar(
                        symbols=["BTCUSDT"],
                        data_type="aggTrades",
                        market_type="futures",
                        futures_type="um",
                        start_date="2024-01-01",
                        days=1,
                        bar_type="time",
                        interval=60_000,
                        use_cache=True,
                    )

                assert not mock_agg_instance.aggregate_time_bars.called

        assert isinstance(result, AggBar)

    def test_different_intervals(self, sample_hive_data):
        """Test that different intervals produce different bar counts."""
        loader = BinanceDataLoader(base_path=sample_hive_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result_1m = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                bar_type="time",
                interval=60_000,
                use_cache=False,
            )

            result_5m = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                bar_type="time",
                interval=300_000,
                use_cache=False,
            )

        assert len(result_1m) > len(result_5m)

    def test_raises_on_no_data(self, sample_hive_data):
        """Test that ValueError is raised when no data is found."""
        loader = BinanceDataLoader(base_path=sample_hive_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            with pytest.raises(ValueError, match="No data found"):
                loader.load_aggbar(
                    symbols=["NONEXISTENT"],
                    data_type="aggTrades",
                    market_type="futures",
                    futures_type="um",
                    start_date="2024-01-01",
                    days=1,
                    bar_type="time",
                    interval=60_000,
                    use_cache=False,
                )


class TestIncrementalDownload:
    def test_find_missing_files_all_exist(self, sample_hive_data):
        loader = BinanceDataLoader(base_path=sample_hive_data)
        missing = loader._find_missing_files(
            symbols=["BTCUSDT", "ETHUSDT"],
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 4),
        )
        assert missing == {}

    def test_find_missing_files_partial_missing(self, sample_hive_data):
        loader = BinanceDataLoader(base_path=sample_hive_data)
        missing = loader._find_missing_files(
            symbols=["BTCUSDT"],
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 10),
        )
        assert "BTCUSDT" in missing
        assert len(missing["BTCUSDT"]) == 6

    def test_find_missing_files_new_symbol(self, sample_hive_data):
        loader = BinanceDataLoader(base_path=sample_hive_data)
        missing = loader._find_missing_files(
            symbols=["BTCUSDT", "NEWCOIN"],
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 4),
        )
        assert "NEWCOIN" in missing
        assert len(missing["NEWCOIN"]) == 3
        assert "BTCUSDT" not in missing

    def test_group_consecutive_dates(self, sample_hive_data):
        loader = BinanceDataLoader(base_path=sample_hive_data)

        dates = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 5), datetime(2024, 1, 6)]
        ranges = loader._group_consecutive_dates(dates)

        assert len(ranges) == 2
        assert ranges[0] == (datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert ranges[1] == (datetime(2024, 1, 5), datetime(2024, 1, 6))

    def test_group_consecutive_dates_empty(self, sample_hive_data):
        loader = BinanceDataLoader(base_path=sample_hive_data)
        assert loader._group_consecutive_dates([]) == []

    def test_download_missing_only_called(self, sample_hive_data):
        loader = BinanceDataLoader(base_path=sample_hive_data)

        with patch.object(loader, "_find_missing_files") as mock_find:
            mock_find.return_value = {"NEWCOIN": [datetime(2024, 1, 5)]}
            with patch.object(loader, "_download_missing_files") as mock_download:
                with patch.object(loader, "_check_all_symbols_exist", return_value=True):
                    loader.load_aggbar(
                        symbols=["BTCUSDT"],
                        data_type="aggTrades",
                        market_type="futures",
                        futures_type="um",
                        start_date="2024-01-01",
                        days=3,
                        bar_type="time",
                        interval=60_000,
                        use_cache=False,
                    )

                mock_find.assert_called_once()
                mock_download.assert_called_once()
