"""Tests for BinanceDataLoader."""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from freezegun import freeze_time

from factorium import BinanceDataLoader, AggBar
from factorium.data import build_hive_path


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_trades_df():
    """Create sample trade data that mimics Binance aggTrades format."""
    np.random.seed(42)
    n_trades = 1000

    # Generate timestamps over 1 hour (in milliseconds)
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
    timestamps = base_ts + np.arange(n_trades) * 3600  # ~3.6 seconds apart

    # Generate prices with random walk
    prices = 100 + np.cumsum(np.random.randn(n_trades) * 0.1)

    # Generate volumes
    volumes = np.abs(np.random.randn(n_trades)) * 10 + 1

    df = pd.DataFrame(
        {
            "agg_trade_id": np.arange(n_trades),
            "price": prices,
            "quantity": volumes,
            "first_trade_id": np.arange(n_trades) * 2,
            "last_trade_id": np.arange(n_trades) * 2 + 1,
            "transact_time": timestamps,
            "is_buyer_maker": np.random.choice([True, False], n_trades),
            "symbol": "BTCUSDT",
        }
    )

    return df


@pytest.fixture
def loader():
    """Create a BinanceDataLoader instance."""
    return BinanceDataLoader(base_path="./test_data")


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def loader_with_temp_dir(temp_data_dir):
    """Create a BinanceDataLoader with temporary directory."""
    return BinanceDataLoader(base_path=str(temp_data_dir))


def create_mock_df(symbol: str, seed: int = 42) -> pd.DataFrame:
    """Create a mock DataFrame for a given symbol."""
    np.random.seed(seed)
    n_trades = 500

    base_ts = 1704067200000
    timestamps = base_ts + np.arange(n_trades) * 3600
    prices = 100 + np.cumsum(np.random.randn(n_trades) * 0.1)
    volumes = np.abs(np.random.randn(n_trades)) * 10 + 1

    return pd.DataFrame(
        {
            "agg_trade_id": np.arange(n_trades),
            "price": prices,
            "quantity": volumes,
            "first_trade_id": np.arange(n_trades) * 2,
            "last_trade_id": np.arange(n_trades) * 2 + 1,
            "transact_time": timestamps,
            "is_buyer_maker": np.random.choice([True, False], n_trades),
            "symbol": symbol,
        }
    )


def create_parquet_file(base_path: Path, market: str, data_type: str, symbol: str, date: datetime) -> Path:
    """Create a test Parquet file in Hive partition format."""
    hive_path = build_hive_path(base_path, market, data_type, symbol, date.year, date.month, date.day)
    hive_path.mkdir(parents=True, exist_ok=True)

    # Create minimal test data
    df = pd.DataFrame(
        {
            "agg_trade_id": [1, 2, 3],
            "price": [100.0, 101.0, 102.0],
            "quantity": [1.0, 2.0, 3.0],
            "transact_time": [
                int(date.timestamp() * 1000),
                int(date.timestamp() * 1000) + 1000,
                int(date.timestamp() * 1000) + 2000,
            ],
            "is_buyer_maker": [True, False, True],
        }
    )

    parquet_path = hive_path / "data.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)

    return parquet_path


# =============================================================================
# TestCalculateDateRange - 日期範圍計算測試
# =============================================================================


class TestCalculateDateRange:
    """Tests for BinanceDataLoader._calculate_date_range method."""

    def test_with_start_and_end_date(self, loader):
        """Test with both start_date and end_date specified."""
        start_dt, end_dt = loader._calculate_date_range(start_date="2024-01-01", end_date="2024-01-07", days=None)

        assert start_dt == datetime(2024, 1, 1)
        assert end_dt == datetime(2024, 1, 7)

    def test_with_start_date_and_days(self, loader):
        """Test with start_date and days specified."""
        start_dt, end_dt = loader._calculate_date_range(start_date="2024-01-01", end_date=None, days=7)

        assert start_dt == datetime(2024, 1, 1)
        assert end_dt == datetime(2024, 1, 8)  # 7 days after start

    @freeze_time("2024-06-15 12:00:00")
    def test_with_only_days(self, loader):
        """Test with only days specified (should use today as end)."""
        start_dt, end_dt = loader._calculate_date_range(start_date=None, end_date=None, days=7)

        assert end_dt == datetime(2024, 6, 15, 12, 0, 0)
        assert start_dt == end_dt - timedelta(days=6)  # days-1 = 6

    @freeze_time("2024-06-15 12:00:00")
    def test_default_7_days(self, loader):
        """Test default behavior (no params = 7 days ending today)."""
        start_dt, end_dt = loader._calculate_date_range(start_date=None, end_date=None, days=None)

        assert end_dt == datetime(2024, 6, 15, 12, 0, 0)
        assert start_dt == end_dt - timedelta(days=6)

    def test_cross_month_range(self, loader):
        """Test date range crossing month boundary."""
        start_dt, end_dt = loader._calculate_date_range(start_date="2024-01-28", end_date=None, days=10)

        assert start_dt == datetime(2024, 1, 28)
        assert end_dt == datetime(2024, 2, 7)  # Crosses into February

    def test_cross_year_range(self, loader):
        """Test date range crossing year boundary."""
        start_dt, end_dt = loader._calculate_date_range(start_date="2023-12-28", end_date="2024-01-05", days=None)

        assert start_dt == datetime(2023, 12, 28)
        assert end_dt == datetime(2024, 1, 5)

    def test_single_day_range(self, loader):
        """Test single day range (start == end)."""
        start_dt, end_dt = loader._calculate_date_range(start_date="2024-01-01", end_date="2024-01-01", days=None)

        assert start_dt == datetime(2024, 1, 1)
        assert end_dt == datetime(2024, 1, 1)

    def test_start_date_with_one_day(self, loader):
        """Test start_date with days=1."""
        start_dt, end_dt = loader._calculate_date_range(start_date="2024-01-01", end_date=None, days=1)

        assert start_dt == datetime(2024, 1, 1)
        assert end_dt == datetime(2024, 1, 2)


# =============================================================================
# TestBuildDateFilter - 日期過濾條件測試
# =============================================================================


# =============================================================================
# TestCheckAllFilesExist - 檔案存在檢查測試
# =============================================================================


class TestCheckAllFilesExist:
    """Tests for BinanceDataLoader._check_all_files_exist method."""

    def test_all_files_exist(self, temp_data_dir):
        """Test when all required files exist."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))

        # Create parquet files for 3 days
        for i in range(3):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            create_parquet_file(temp_data_dir, "futures_um", "aggTrades", "BTCUSDT", date)

        result = loader._check_all_files_exist(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 4),  # end is exclusive
        )

        assert result is True

    def test_some_files_missing(self, temp_data_dir):
        """Test when some files are missing."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))

        # Create files for day 1 and 3, skip day 2
        create_parquet_file(temp_data_dir, "futures_um", "aggTrades", "BTCUSDT", datetime(2024, 1, 1))
        create_parquet_file(temp_data_dir, "futures_um", "aggTrades", "BTCUSDT", datetime(2024, 1, 3))

        result = loader._check_all_files_exist(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 4),
        )

        assert result is False

    def test_no_files_exist(self, temp_data_dir):
        """Test when no files exist."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))

        result = loader._check_all_files_exist(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 4),
        )

        assert result is False

    def test_checks_correct_path_structure(self, temp_data_dir):
        """Test that check uses correct Hive partition path."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))

        # Create file with correct structure
        expected_path = (
            temp_data_dir
            / "market=futures_um"
            / "data_type=aggTrades"
            / "symbol=BTCUSDT"
            / "year=2024"
            / "month=01"
            / "day=01"
            / "data.parquet"
        )
        expected_path.parent.mkdir(parents=True, exist_ok=True)

        # Create minimal parquet
        df = pd.DataFrame({"a": [1]})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, expected_path)

        result = loader._check_all_files_exist(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 2),
        )

        assert result is True

    def test_spot_market_path(self, temp_data_dir):
        """Test file check for spot market (different path)."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))

        # Create file for spot market
        expected_path = (
            temp_data_dir
            / "market=spot"
            / "data_type=trades"
            / "symbol=BTCUSDT"
            / "year=2024"
            / "month=01"
            / "day=01"
            / "data.parquet"
        )
        expected_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({"a": [1]})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, expected_path)

        result = loader._check_all_files_exist(
            symbol="BTCUSDT",
            data_type="trades",
            market_type="spot",
            futures_type="",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 2),
        )

        assert result is True


# =============================================================================
# TestLoadAggbar - AggBar 載入測試
# =============================================================================


class TestLoadAggbar:
    """Tests for BinanceDataLoader.load_aggbar method.

    These are integration tests that use real Parquet files in temp directories.
    """

    def test_load_aggbar_single_symbol(self, loader_with_temp_dir, temp_data_dir):
        """Test load_aggbar with a single symbol."""
        # Create test parquet files
        for i in range(3):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            _create_test_parquet_file(temp_data_dir, "BTCUSDT", date, n_trades=100)

        agg = loader_with_temp_dir.load_aggbar(
            symbols=["BTCUSDT"],
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_date="2024-01-01",
            days=3,
            bar_type="time",
            interval=60_000,
            use_cache=False,  # Disable cache for test isolation
        )

        assert isinstance(agg, AggBar)
        assert len(agg.symbols) == 1
        assert "BTCUSDT" in agg.symbols
        assert len(agg) > 0

    def test_load_aggbar_multiple_symbols(self, loader_with_temp_dir, temp_data_dir):
        """Test load_aggbar with multiple symbols."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        # Create test parquet files for each symbol
        for symbol in symbols:
            for i in range(2):
                date = datetime(2024, 1, 1) + timedelta(days=i)
                _create_test_parquet_file(temp_data_dir, symbol, date, n_trades=50)

        agg = loader_with_temp_dir.load_aggbar(
            symbols=symbols,
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_date="2024-01-01",
            days=2,
            bar_type="time",
            interval=60_000,
            use_cache=False,
        )

        assert isinstance(agg, AggBar)
        assert len(agg.symbols) == 3
        for symbol in symbols:
            assert symbol in agg.symbols

    def test_load_aggbar_returns_valid_aggbar(self, loader_with_temp_dir, temp_data_dir):
        """Test that load_aggbar returns an AggBar with correct structure."""
        _create_test_parquet_file(temp_data_dir, "BTCUSDT", datetime(2024, 1, 1), n_trades=100)

        agg = loader_with_temp_dir.load_aggbar(
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

        # Check AggBar has expected columns
        expected_cols = ["start_time", "end_time", "symbol", "open", "high", "low", "close", "volume"]
        for col in expected_cols:
            assert col in agg.cols, f"Missing column: {col}"

    def test_load_aggbar_can_extract_factors(self, loader_with_temp_dir, temp_data_dir):
        """Test that factors can be extracted from the returned AggBar."""
        _create_test_parquet_file(temp_data_dir, "BTCUSDT", datetime(2024, 1, 1), n_trades=200)

        agg = loader_with_temp_dir.load_aggbar(
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

        # Extract factor
        close = agg["close"]
        assert close.name == "close"
        assert len(close) == len(agg)

        # Perform operations on factor
        momentum = close.ts_delta(5)
        assert len(momentum) == len(close)

    def test_load_aggbar_different_intervals(self, loader_with_temp_dir, temp_data_dir):
        """Test that different intervals produce different bar counts."""
        _create_test_parquet_file(temp_data_dir, "BTCUSDT", datetime(2024, 1, 1), n_trades=500)

        # 1-minute bars
        agg_1min = loader_with_temp_dir.load_aggbar(
            symbols=["BTCUSDT"],
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_date="2024-01-01",
            days=1,
            bar_type="time",
            interval=60_000,  # 1 minute
            use_cache=False,
        )

        # 5-minute bars
        agg_5min = loader_with_temp_dir.load_aggbar(
            symbols=["BTCUSDT"],
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_date="2024-01-01",
            days=1,
            bar_type="time",
            interval=300_000,  # 5 minutes
            use_cache=False,
        )

        # 5-minute bars should have fewer rows than 1-minute bars
        assert len(agg_5min) < len(agg_1min)

    def test_load_aggbar_preserves_data_integrity(self, loader_with_temp_dir, temp_data_dir):
        """Test that data from multiple symbols is correctly preserved."""
        # Create data with distinct price ranges for each symbol
        _create_test_parquet_file(temp_data_dir, "BTCUSDT", datetime(2024, 1, 1), n_trades=100, base_price=50000.0)
        _create_test_parquet_file(temp_data_dir, "ETHUSDT", datetime(2024, 1, 1), n_trades=100, base_price=3000.0)

        agg = loader_with_temp_dir.load_aggbar(
            symbols=["BTCUSDT", "ETHUSDT"],
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_date="2024-01-01",
            days=1,
            bar_type="time",
            interval=60_000,
            use_cache=False,
        )

        # Check that prices are distinct for each symbol
        btc_data = agg.data[agg.data["symbol"] == "BTCUSDT"]
        eth_data = agg.data[agg.data["symbol"] == "ETHUSDT"]

        # BTC prices should be much higher than ETH
        assert btc_data["close"].mean() > eth_data["close"].mean() * 5


def _create_test_parquet_file(
    base_path: Path, symbol: str, date: datetime, n_trades: int = 100, base_price: float = 100.0
) -> Path:
    """Create a test Parquet file with realistic trade data.

    Args:
        base_path: Base directory for data storage
        symbol: Trading symbol
        date: Date for the data
        n_trades: Number of trades to generate
        base_price: Base price for the symbol

    Returns:
        Path to the created parquet file
    """
    hive_path = build_hive_path(base_path, "futures_um", "aggTrades", symbol, date.year, date.month, date.day)
    hive_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamps spread across the day
    base_ts = int(date.timestamp() * 1000)
    timestamps = base_ts + np.arange(n_trades) * (86400000 // n_trades)  # Spread evenly across day

    # Generate price data with small variations
    np.random.seed(hash(f"{symbol}_{date}") % 2**32)
    prices = base_price + np.cumsum(np.random.randn(n_trades) * (base_price * 0.0001))
    volumes = np.abs(np.random.randn(n_trades)) * 10 + 1

    df = pd.DataFrame(
        {
            "agg_trade_id": np.arange(n_trades) + int(date.timestamp()),
            "price": prices,
            "quantity": volumes,
            "first_trade_id": np.arange(n_trades) * 2,
            "last_trade_id": np.arange(n_trades) * 2 + 1,
            "transact_time": timestamps,
            "is_buyer_maker": np.random.choice([True, False], n_trades),
        }
    )

    file_path = hive_path / "data.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)

    return file_path


# =============================================================================
# TestLoadAggbarEdgeCases - AggBar 邊界條件測試
# =============================================================================


class TestLoadAggbarEdgeCases:
    """Edge case tests for load_aggbar."""

    def test_load_aggbar_empty_symbols_list(self, loader_with_temp_dir):
        """Test load_aggbar with empty symbols list raises ValueError."""
        with pytest.raises((ValueError, Exception)):  # Can raise ValueError or DuckDB ParserException
            loader_with_temp_dir.load_aggbar(
                symbols=[],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                bar_type="time",
                interval=60_000,
            )

    def test_non_time_bar_with_multiple_symbols_raises(self, loader_with_temp_dir):
        """Test that non-time bars with multiple symbols raises ValueError."""
        with pytest.raises(ValueError, match="only supports single symbol"):
            loader_with_temp_dir.load_aggbar(
                symbols=["BTCUSDT", "ETHUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                bar_type="tick",  # Non-time bar
                interval=100,
            )


# =============================================================================
# TestLoaderInitialization - Loader 初始化測試
# =============================================================================


class TestLoaderInitialization:
    """Tests for BinanceDataLoader initialization."""

    def test_default_base_path(self):
        """Test default base_path is './Data'."""
        loader = BinanceDataLoader()
        assert loader.base_path == Path("./Data")

    def test_custom_base_path(self, temp_data_dir):
        """Test custom base_path is set correctly."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        assert loader.base_path == temp_data_dir

    def test_downloader_is_created(self, temp_data_dir):
        """Test that BinanceDataDownloader is created."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        assert loader.downloader is not None

    def test_download_settings_passed_to_downloader(self, temp_data_dir):
        """Test that download settings are passed to downloader."""
        loader = BinanceDataLoader(
            base_path=str(temp_data_dir), max_concurrent_downloads=10, retry_attempts=5, retry_delay=2
        )

        assert loader.downloader.max_concurrent_downloads == 10
        assert loader.downloader.retry_attempts == 5
        assert loader.downloader.retry_delay == 2
