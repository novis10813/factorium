"""Tests for BarCache."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorium.data.cache import BarCache
from factorium.storage import LocalStorageBackend


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache(temp_cache_dir):
    """Create cache using non-deprecated storage API."""
    storage = LocalStorageBackend(str(temp_cache_dir))
    return BarCache(storage=storage, cache_prefix="")


@pytest.fixture
def sample_bar_df():
    """Create sample bar DataFrame in Polars format."""
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 50 + ["ETHUSDT"] * 50,
            "start_time": [1704067200000 + i * 60000 for i in range(50)] * 2,
            "end_time": [1704067260000 + i * 60000 for i in range(50)] * 2,
            "open": [np.random.randn() + 100 for _ in range(100)],
            "high": [np.random.randn() + 101 for _ in range(100)],
            "low": [np.random.randn() + 99 for _ in range(100)],
            "close": [np.random.randn() + 100 for _ in range(100)],
            "volume": [abs(np.random.randn()) * 100 for _ in range(100)],
            "vwap": [np.random.randn() + 100 for _ in range(100)],
        }
    )


class TestBarCache:
    """Tests for BarCache."""

    def test_cache_initialization(self, temp_cache_dir, cache):
        """Test cache initializes correctly."""
        assert cache.cache_dir is None
        assert temp_cache_dir.exists()

    def test_cache_miss_returns_none(self, cache):
        """Test that cache miss returns None."""
        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )

        assert result is None

    def test_cache_put_and_get(self, cache, sample_bar_df):
        """Test putting and getting from cache."""
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT", "ETHUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )

        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT", "ETHUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )

        assert result is not None
        assert len(result) == len(sample_bar_df)

    def test_cache_key_different_symbols(self, cache, sample_bar_df):
        """Test that different symbols produce different cache keys."""
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )

        result = cache.get(
            exchange="binance",
            symbols=["ETHUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )

        assert result is None

    def test_cache_key_different_interval(self, cache, sample_bar_df):
        """Test that different intervals produce different cache keys."""
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )

        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=300_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )

        assert result is None

    def test_cache_daily_files(self, temp_cache_dir, cache, sample_bar_df):
        """Test that cache creates daily files."""
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 15),
        )

        cache_files = list(temp_cache_dir.rglob("*.parquet"))
        assert len(cache_files) == 1
        assert "2024-01-15" in cache_files[0].name

    def test_get_date_range(self, cache, sample_bar_df):
        """Test getting data for a date range."""
        for day in range(1, 4):
            cache.put(
                df=sample_bar_df,
                exchange="binance",
                symbols=["BTCUSDT"],
                interval_ms=60_000,
                data_type="aggTrades",
                market_type="futures",
                date=datetime(2024, 1, day),
            )

        result = cache.get_range(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
        )

        assert result is not None
        assert len(result) == len(sample_bar_df) * 3

    def test_get_date_range_partial_miss(self, cache, sample_bar_df):
        """Test that partial cache miss returns None for range."""
        for day in [1, 3]:
            cache.put(
                df=sample_bar_df,
                exchange="binance",
                symbols=["BTCUSDT"],
                interval_ms=60_000,
                data_type="aggTrades",
                market_type="futures",
                date=datetime(2024, 1, day),
            )

        result = cache.get_range(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
        )

        assert result is None

    def test_clear_cache(self, cache, sample_bar_df):
        """Test clearing the cache."""
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )

        cleared = cache.clear()
        assert cleared >= 1

        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        assert result is None
