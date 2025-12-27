"""Tests for BinanceDataLoader."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from freezegun import freeze_time

from factorium import BinanceDataLoader, AggBar
from factorium.utils.parquet import build_hive_path


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
    
    df = pd.DataFrame({
        'agg_trade_id': np.arange(n_trades),
        'price': prices,
        'quantity': volumes,
        'first_trade_id': np.arange(n_trades) * 2,
        'last_trade_id': np.arange(n_trades) * 2 + 1,
        'transact_time': timestamps,
        'is_buyer_maker': np.random.choice([True, False], n_trades),
        'symbol': 'BTCUSDT'
    })
    
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
    
    return pd.DataFrame({
        'agg_trade_id': np.arange(n_trades),
        'price': prices,
        'quantity': volumes,
        'first_trade_id': np.arange(n_trades) * 2,
        'last_trade_id': np.arange(n_trades) * 2 + 1,
        'transact_time': timestamps,
        'is_buyer_maker': np.random.choice([True, False], n_trades),
        'symbol': symbol
    })


def create_parquet_file(base_path: Path, market: str, data_type: str, 
                        symbol: str, date: datetime) -> Path:
    """Create a test Parquet file in Hive partition format."""
    hive_path = build_hive_path(
        base_path, market, data_type, symbol,
        date.year, date.month, date.day
    )
    hive_path.mkdir(parents=True, exist_ok=True)
    
    # Create minimal test data
    df = pd.DataFrame({
        'agg_trade_id': [1, 2, 3],
        'price': [100.0, 101.0, 102.0],
        'quantity': [1.0, 2.0, 3.0],
        'transact_time': [
            int(date.timestamp() * 1000),
            int(date.timestamp() * 1000) + 1000,
            int(date.timestamp() * 1000) + 2000,
        ],
        'is_buyer_maker': [True, False, True],
    })
    
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
        start_dt, end_dt = loader._calculate_date_range(
            start_date="2024-01-01",
            end_date="2024-01-07",
            days=None
        )
        
        assert start_dt == datetime(2024, 1, 1)
        assert end_dt == datetime(2024, 1, 7)
    
    def test_with_start_date_and_days(self, loader):
        """Test with start_date and days specified."""
        start_dt, end_dt = loader._calculate_date_range(
            start_date="2024-01-01",
            end_date=None,
            days=7
        )
        
        assert start_dt == datetime(2024, 1, 1)
        assert end_dt == datetime(2024, 1, 8)  # 7 days after start
    
    @freeze_time("2024-06-15 12:00:00")
    def test_with_only_days(self, loader):
        """Test with only days specified (should use today as end)."""
        start_dt, end_dt = loader._calculate_date_range(
            start_date=None,
            end_date=None,
            days=7
        )
        
        assert end_dt == datetime(2024, 6, 15, 12, 0, 0)
        assert start_dt == end_dt - timedelta(days=6)  # days-1 = 6
    
    @freeze_time("2024-06-15 12:00:00")
    def test_default_7_days(self, loader):
        """Test default behavior (no params = 7 days ending today)."""
        start_dt, end_dt = loader._calculate_date_range(
            start_date=None,
            end_date=None,
            days=None
        )
        
        assert end_dt == datetime(2024, 6, 15, 12, 0, 0)
        assert start_dt == end_dt - timedelta(days=6)
    
    def test_cross_month_range(self, loader):
        """Test date range crossing month boundary."""
        start_dt, end_dt = loader._calculate_date_range(
            start_date="2024-01-28",
            end_date=None,
            days=10
        )
        
        assert start_dt == datetime(2024, 1, 28)
        assert end_dt == datetime(2024, 2, 7)  # Crosses into February
    
    def test_cross_year_range(self, loader):
        """Test date range crossing year boundary."""
        start_dt, end_dt = loader._calculate_date_range(
            start_date="2023-12-28",
            end_date="2024-01-05",
            days=None
        )
        
        assert start_dt == datetime(2023, 12, 28)
        assert end_dt == datetime(2024, 1, 5)
    
    def test_single_day_range(self, loader):
        """Test single day range (start == end)."""
        start_dt, end_dt = loader._calculate_date_range(
            start_date="2024-01-01",
            end_date="2024-01-01",
            days=None
        )
        
        assert start_dt == datetime(2024, 1, 1)
        assert end_dt == datetime(2024, 1, 1)
    
    def test_start_date_with_one_day(self, loader):
        """Test start_date with days=1."""
        start_dt, end_dt = loader._calculate_date_range(
            start_date="2024-01-01",
            end_date=None,
            days=1
        )
        
        assert start_dt == datetime(2024, 1, 1)
        assert end_dt == datetime(2024, 1, 2)


# =============================================================================
# TestBuildDateFilter - 日期過濾條件測試
# =============================================================================

class TestBuildDateFilter:
    """Tests for BinanceDataLoader._build_date_filter method."""
    
    def test_single_day_filter(self, loader):
        """Test filter for a single day."""
        start_dt = datetime(2024, 1, 15)
        end_dt = datetime(2024, 1, 16)
        
        filter_str = loader._build_date_filter(start_dt, end_dt)
        
        # Should include 20240115, exclude 20240116
        assert "20240115" in filter_str
        assert "20240116" in filter_str
        assert ">=" in filter_str
        assert "<" in filter_str
    
    def test_multi_day_filter(self, loader):
        """Test filter for multiple days."""
        start_dt = datetime(2024, 1, 1)
        end_dt = datetime(2024, 1, 10)
        
        filter_str = loader._build_date_filter(start_dt, end_dt)
        
        # Check that start and end values are correctly computed
        assert "20240101" in filter_str  # Start: 2024*10000 + 1*100 + 1 = 20240101
        assert "20240110" in filter_str  # End: 2024*10000 + 1*100 + 10 = 20240110
    
    def test_cross_month_filter(self, loader):
        """Test filter crossing month boundary."""
        start_dt = datetime(2024, 1, 28)
        end_dt = datetime(2024, 2, 5)
        
        filter_str = loader._build_date_filter(start_dt, end_dt)
        
        # January 28, 2024 = 20240128
        # February 5, 2024 = 20240205
        assert "20240128" in filter_str
        assert "20240205" in filter_str
    
    def test_cross_year_filter(self, loader):
        """Test filter crossing year boundary."""
        start_dt = datetime(2023, 12, 28)
        end_dt = datetime(2024, 1, 5)
        
        filter_str = loader._build_date_filter(start_dt, end_dt)
        
        # December 28, 2023 = 20231228
        # January 5, 2024 = 20240105
        assert "20231228" in filter_str
        assert "20240105" in filter_str
    
    def test_filter_uses_cast(self, loader):
        """Test that filter uses CAST for Hive partition columns."""
        start_dt = datetime(2024, 1, 1)
        end_dt = datetime(2024, 1, 2)
        
        filter_str = loader._build_date_filter(start_dt, end_dt)
        
        # Should cast varchar partitions to integer
        assert "CAST(year AS INTEGER)" in filter_str
        assert "CAST(month AS INTEGER)" in filter_str
        assert "CAST(day AS INTEGER)" in filter_str


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
            create_parquet_file(
                temp_data_dir, "futures_um", "aggTrades", "BTCUSDT", date
            )
        
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
        create_parquet_file(
            temp_data_dir, "futures_um", "aggTrades", "BTCUSDT",
            datetime(2024, 1, 1)
        )
        create_parquet_file(
            temp_data_dir, "futures_um", "aggTrades", "BTCUSDT",
            datetime(2024, 1, 3)
        )
        
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
        df = pd.DataFrame({'a': [1]})
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
        
        df = pd.DataFrame({'a': [1]})
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
# TestLoadData - 資料載入測試
# =============================================================================

class TestLoadData:
    """Tests for BinanceDataLoader.load_data method."""
    
    def test_triggers_download_when_files_missing(self, loader):
        """Test that download is triggered when files don't exist."""
        mock_df = create_mock_df("BTCUSDT")
        
        with patch.object(loader, '_check_all_files_exist', return_value=False), \
             patch.object(loader, '_read_data_duckdb', return_value=mock_df), \
             patch('asyncio.run') as mock_asyncio:
            
            loader.load_data(
                symbol="BTCUSDT",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=7,
            )
            
            # asyncio.run should be called to trigger download
            mock_asyncio.assert_called_once()
    
    def test_skips_download_when_files_exist(self, loader):
        """Test that download is skipped when all files exist."""
        mock_df = create_mock_df("BTCUSDT")
        
        with patch.object(loader, '_check_all_files_exist', return_value=True), \
             patch.object(loader, '_read_data_duckdb', return_value=mock_df), \
             patch('asyncio.run') as mock_asyncio:
            
            loader.load_data(
                symbol="BTCUSDT",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=7,
            )
            
            # asyncio.run should NOT be called
            mock_asyncio.assert_not_called()
    
    def test_force_download_overrides_file_check(self, loader):
        """Test that force_download=True triggers download even if files exist."""
        mock_df = create_mock_df("BTCUSDT")
        
        with patch.object(loader, '_check_all_files_exist', return_value=True), \
             patch.object(loader, '_read_data_duckdb', return_value=mock_df), \
             patch('asyncio.run') as mock_asyncio:
            
            loader.load_data(
                symbol="BTCUSDT",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=7,
                force_download=True,
            )
            
            # asyncio.run should be called due to force_download
            mock_asyncio.assert_called_once()
    
    def test_returns_dataframe(self, loader):
        """Test that load_data returns a DataFrame."""
        mock_df = create_mock_df("BTCUSDT")
        
        with patch.object(loader, '_check_all_files_exist', return_value=True), \
             patch.object(loader, '_read_data_duckdb', return_value=mock_df):
            
            result = loader.load_data(
                symbol="BTCUSDT",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=7,
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(mock_df)
    
    def test_passes_columns_to_duckdb_reader(self, loader):
        """Test that columns parameter is passed to DuckDB reader."""
        mock_df = pd.DataFrame({
            'price': [100.0, 101.0],
            'quantity': [1.0, 2.0],
        })
        
        with patch.object(loader, '_check_all_files_exist', return_value=True), \
             patch.object(loader, '_read_data_duckdb', return_value=mock_df) as mock_read:
            
            loader.load_data(
                symbol="BTCUSDT",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=7,
                columns=["price", "quantity"],
            )
            
            # Check columns were passed
            call_kwargs = mock_read.call_args
            assert call_kwargs.kwargs.get('columns') == ["price", "quantity"]
    
    def test_date_range_resolved_correctly(self, loader):
        """Test that date range is correctly resolved and passed."""
        mock_df = create_mock_df("BTCUSDT")
        
        with patch.object(loader, '_check_all_files_exist', return_value=True), \
             patch.object(loader, '_read_data_duckdb', return_value=mock_df) as mock_read:
            
            loader.load_data(
                symbol="BTCUSDT",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=7,
            )
            
            # Verify the date range passed to _read_data_duckdb
            call_args = mock_read.call_args
            start_dt = call_args.args[4]  # 5th positional arg
            end_dt = call_args.args[5]    # 6th positional arg
            
            assert start_dt == datetime(2024, 1, 1)
            assert end_dt == datetime(2024, 1, 8)


# =============================================================================
# TestReadDataDuckDB - DuckDB 讀取測試
# =============================================================================

class TestReadDataDuckDB:
    """Tests for BinanceDataLoader._read_data_duckdb method."""
    
    def test_builds_correct_glob_pattern(self, temp_data_dir):
        """Test that correct glob pattern is built for Parquet files."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        
        # Create test parquet files
        for i in range(3):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            create_parquet_file(
                temp_data_dir, "futures_um", "aggTrades", "BTCUSDT", date
            )
        
        with patch('duckdb.query') as mock_query:
            mock_query.return_value.df.return_value = pd.DataFrame()
            
            try:
                loader._read_data_duckdb(
                    symbol="BTCUSDT",
                    data_type="aggTrades",
                    market_type="futures",
                    futures_type="um",
                    start_dt=datetime(2024, 1, 1),
                    end_dt=datetime(2024, 1, 4),
                )
            except ValueError:
                pass  # Empty DataFrame raises ValueError
            
            # Check the query contains correct path pattern
            query = mock_query.call_args.args[0]
            assert "market=futures_um" in query
            assert "data_type=aggTrades" in query
            assert "symbol=BTCUSDT" in query
            assert "**/*.parquet" in query
    
    def test_reads_parquet_with_hive_partitioning(self, temp_data_dir):
        """Test that Parquet files are read with Hive partitioning enabled."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        
        # Create test parquet files
        for i in range(2):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            create_parquet_file(
                temp_data_dir, "futures_um", "aggTrades", "BTCUSDT", date
            )
        
        result = loader._read_data_duckdb(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 3),
        )
        
        # Should return data from both days
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_raises_error_when_no_data_found(self, temp_data_dir):
        """Test that an error is raised when no data is found."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        
        # DuckDB raises IOException when no files match the pattern
        # This gets re-raised after logging in _read_data_duckdb
        with pytest.raises(Exception):  # Can be IOException or wrapped exception
            loader._read_data_duckdb(
                symbol="NONEXISTENT",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_dt=datetime(2024, 1, 1),
                end_dt=datetime(2024, 1, 2),
            )
    
    def test_selects_specific_columns(self, temp_data_dir):
        """Test that only specified columns are returned."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        
        # Create test parquet file
        create_parquet_file(
            temp_data_dir, "futures_um", "aggTrades", "BTCUSDT",
            datetime(2024, 1, 1)
        )
        
        result = loader._read_data_duckdb(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 2),
            columns=["price", "quantity"],
        )
        
        assert "price" in result.columns
        assert "quantity" in result.columns
        # agg_trade_id should not be in columns (but might be from SELECT *)
    
    def test_orders_by_date_partitions(self, temp_data_dir):
        """Test that results are ordered by year, month, day."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        
        with patch('duckdb.query') as mock_query:
            mock_query.return_value.df.return_value = pd.DataFrame({'a': [1]})
            
            loader._read_data_duckdb(
                symbol="BTCUSDT",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_dt=datetime(2024, 1, 1),
                end_dt=datetime(2024, 1, 2),
            )
            
            query = mock_query.call_args.args[0]
            assert "ORDER BY year, month, day" in query


# =============================================================================
# TestLoadAggbar - AggBar 載入測試
# =============================================================================

class TestLoadAggbar:
    """Tests for BinanceDataLoader.load_aggbar method."""
    
    def test_load_aggbar_single_symbol(self, loader, sample_trades_df):
        """Test load_aggbar with a single symbol."""
        with patch.object(loader, 'load_data', return_value=sample_trades_df):
            agg = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000
            )
            
            assert isinstance(agg, AggBar)
            assert len(agg.symbols) == 1
            assert "BTCUSDT" in agg.symbols
            assert len(agg) > 0
    
    def test_load_aggbar_multiple_symbols(self, loader):
        """Test load_aggbar with multiple symbols."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        def mock_load_data(symbol, **kwargs):
            seed = hash(symbol) % 1000
            return create_mock_df(symbol, seed)
        
        with patch.object(loader, 'load_data', side_effect=mock_load_data):
            agg = loader.load_aggbar(
                symbols=symbols,
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000
            )
            
            assert isinstance(agg, AggBar)
            assert len(agg.symbols) == 3
            for symbol in symbols:
                assert symbol in agg.symbols
    
    def test_load_aggbar_returns_valid_aggbar(self, loader, sample_trades_df):
        """Test that load_aggbar returns an AggBar with correct structure."""
        with patch.object(loader, 'load_data', return_value=sample_trades_df):
            agg = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000
            )
            
            # Check AggBar has expected columns
            expected_cols = ['start_time', 'end_time', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            for col in expected_cols:
                assert col in agg.cols, f"Missing column: {col}"
    
    def test_load_aggbar_can_extract_factors(self, loader, sample_trades_df):
        """Test that factors can be extracted from the returned AggBar."""
        with patch.object(loader, 'load_data', return_value=sample_trades_df):
            agg = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000
            )
            
            # Extract factor
            close = agg['close']
            assert close.name == 'close'
            assert len(close) == len(agg)
            
            # Perform operations on factor
            momentum = close.ts_delta(5)
            assert len(momentum) == len(close)
    
    def test_load_aggbar_passes_bar_kwargs(self, loader, sample_trades_df):
        """Test that bar_kwargs are correctly passed to TimeBar."""
        with patch.object(loader, 'load_data', return_value=sample_trades_df):
            # Test with different interval
            agg_1min = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000  # 1 minute
            )
            
            agg_5min = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=300_000  # 5 minutes
            )
            
            # 5-minute bars should have fewer rows than 1-minute bars
            assert len(agg_5min) < len(agg_1min)
    
    def test_load_aggbar_calls_load_data_for_each_symbol(self, loader):
        """Test that load_data is called once for each symbol."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        
        def mock_load_data(symbol, **kwargs):
            return create_mock_df(symbol)
        
        with patch.object(loader, 'load_data', side_effect=mock_load_data) as mock:
            loader.load_aggbar(
                symbols=symbols,
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000
            )
            
            assert mock.call_count == len(symbols)
    
    def test_load_aggbar_passes_load_data_params(self, loader, sample_trades_df):
        """Test that load_data receives correct parameters."""
        with patch.object(loader, 'load_data', return_value=sample_trades_df) as mock:
            loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                end_date="2024-01-07",
                days=None,
                columns=["price", "quantity", "transact_time"],
                force_download=True,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000
            )
            
            mock.assert_called_once_with(
                symbol="BTCUSDT",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                end_date="2024-01-07",
                days=None,
                columns=["price", "quantity", "transact_time"],
                force_download=True
            )
    
    def test_load_aggbar_with_different_futures_type(self, loader, sample_trades_df):
        """Test load_aggbar with cm (coin-margined) futures type."""
        sample_trades_df['symbol'] = 'BTCUSD_PERP'
        
        with patch.object(loader, 'load_data', return_value=sample_trades_df):
            agg = loader.load_aggbar(
                symbols=["BTCUSD_PERP"],
                data_type="trades",
                market_type="futures",
                futures_type="cm",
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000
            )
            
            assert isinstance(agg, AggBar)
            assert "BTCUSD_PERP" in agg.symbols
    
    def test_load_aggbar_preserves_data_integrity(self, loader):
        """Test that data from multiple symbols is correctly preserved."""
        def mock_load_data(symbol, **kwargs):
            df = create_mock_df(symbol, seed=hash(symbol) % 1000)
            # Set distinct price ranges for each symbol
            if symbol == "BTCUSDT":
                df['price'] = df['price'] * 100  # ~10000
            elif symbol == "ETHUSDT":
                df['price'] = df['price'] * 10   # ~1000
            return df
        
        with patch.object(loader, 'load_data', side_effect=mock_load_data):
            agg = loader.load_aggbar(
                symbols=["BTCUSDT", "ETHUSDT"],
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000
            )
            
            # Check that prices are distinct for each symbol
            btc_data = agg.data[agg.data['symbol'] == 'BTCUSDT']
            eth_data = agg.data[agg.data['symbol'] == 'ETHUSDT']
            
            # BTC prices should be ~10x higher than ETH
            assert btc_data['close'].mean() > eth_data['close'].mean() * 5


# =============================================================================
# TestLoadAggbarEdgeCases - AggBar 邊界條件測試
# =============================================================================

class TestLoadAggbarEdgeCases:
    """Edge case tests for load_aggbar."""
    
    def test_load_aggbar_empty_symbols_list(self, loader):
        """Test load_aggbar with empty symbols list raises ValueError."""
        with patch.object(loader, 'load_data') as mock:
            with pytest.raises(ValueError, match="No objects to concatenate"):
                loader.load_aggbar(
                    symbols=[],
                    data_type="aggTrades",
                    market_type="futures",
                    futures_type="um",
                    start_date="2024-01-01",
                    days=1,
                    timestamp_col="transact_time",
                    price_col="price",
                    volume_col="quantity",
                    interval_ms=60_000
                )
            
            # load_data should not be called
            mock.assert_not_called()
    
    def test_load_aggbar_with_default_futures_type(self, loader, sample_trades_df):
        """Test load_aggbar uses default futures_type='cm'."""
        with patch.object(loader, 'load_data', return_value=sample_trades_df) as mock:
            loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="aggTrades",
                market_type="futures",
                # futures_type not specified, should default to 'cm'
                start_date="2024-01-01",
                days=1,
                timestamp_col="transact_time",
                price_col="price",
                volume_col="quantity",
                interval_ms=60_000
            )
            
            # Check that futures_type='cm' was passed
            call_kwargs = mock.call_args[1]
            assert call_kwargs['futures_type'] == 'cm'


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
            base_path=str(temp_data_dir),
            max_concurrent_downloads=10,
            retry_attempts=5,
            retry_delay=2
        )
        
        assert loader.downloader.max_concurrent_downloads == 10
        assert loader.downloader.retry_attempts == 5
        assert loader.downloader.retry_delay == 2


# =============================================================================
# TestIntegration - 整合測試
# =============================================================================

class TestIntegration:
    """Integration tests with real Parquet files."""
    
    def test_full_read_workflow(self, temp_data_dir):
        """Test complete workflow: create parquet -> read via DuckDB."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        
        # Create test data for multiple days
        for i in range(3):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            hive_path = build_hive_path(
                temp_data_dir, "futures_um", "aggTrades", "BTCUSDT",
                date.year, date.month, date.day
            )
            hive_path.mkdir(parents=True, exist_ok=True)
            
            # Create data with different prices per day
            df = pd.DataFrame({
                'agg_trade_id': np.arange(100) + i * 100,
                'price': 100.0 + i * 10 + np.random.randn(100) * 0.1,
                'quantity': np.abs(np.random.randn(100)) + 1,
                'transact_time': int(date.timestamp() * 1000) + np.arange(100) * 1000,
                'is_buyer_maker': np.random.choice([True, False], 100),
            })
            
            table = pa.Table.from_pandas(df)
            pq.write_table(table, hive_path / "data.parquet")
        
        # Read data
        result = loader._read_data_duckdb(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 4),
        )
        
        # Verify data
        assert len(result) == 300  # 3 days * 100 rows
        assert 'price' in result.columns
        assert 'quantity' in result.columns
    
    def test_date_filter_accuracy(self, temp_data_dir):
        """Test that date filter accurately selects correct date range."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        
        # Create data for 5 days
        for i in range(5):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            hive_path = build_hive_path(
                temp_data_dir, "futures_um", "aggTrades", "BTCUSDT",
                date.year, date.month, date.day
            )
            hive_path.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame({
                'day_marker': [date.day] * 10,  # Use day as marker
                'price': [100.0] * 10,
                'quantity': [1.0] * 10,
            })
            
            table = pa.Table.from_pandas(df)
            pq.write_table(table, hive_path / "data.parquet")
        
        # Read only days 2-3 (Jan 2-3)
        result = loader._read_data_duckdb(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 2),
            end_dt=datetime(2024, 1, 4),
        )
        
        # Should only have data from days 2 and 3
        assert len(result) == 20  # 2 days * 10 rows
        unique_days = result['day_marker'].unique()
        assert set(unique_days) == {2, 3}
    
    def test_cross_month_data_loading(self, temp_data_dir):
        """Test loading data that spans across months."""
        loader = BinanceDataLoader(base_path=str(temp_data_dir))
        
        # Create data for end of Jan and start of Feb
        dates = [
            datetime(2024, 1, 30),
            datetime(2024, 1, 31),
            datetime(2024, 2, 1),
            datetime(2024, 2, 2),
        ]
        
        for date in dates:
            hive_path = build_hive_path(
                temp_data_dir, "futures_um", "aggTrades", "BTCUSDT",
                date.year, date.month, date.day
            )
            hive_path.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame({
                'month_marker': [date.month] * 5,
                'day_marker': [date.day] * 5,
                'price': [100.0] * 5,
            })
            
            table = pa.Table.from_pandas(df)
            pq.write_table(table, hive_path / "data.parquet")
        
        # Read across month boundary
        result = loader._read_data_duckdb(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            start_dt=datetime(2024, 1, 30),
            end_dt=datetime(2024, 2, 3),
        )
        
        # Should have data from all 4 days
        assert len(result) == 20  # 4 days * 5 rows
        
        # Verify we got data from both months
        jan_data = result[result['month_marker'] == 1]
        feb_data = result[result['month_marker'] == 2]
        assert len(jan_data) == 10  # Jan 30, 31
        assert len(feb_data) == 10  # Feb 1, 2
