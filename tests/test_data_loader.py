"""Tests for BinanceDataLoader."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from factorium import BinanceDataLoader, AggBar


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
