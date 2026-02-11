"""Tests for BarCache with Polars DataFrames."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from factorium.data.cache import BarCache


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_bar_df_polars():
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


class TestBarCachePolars:
    """Tests for BarCache with Polars DataFrames."""

    def test_put_and_get_polars(self, temp_cache_dir, sample_bar_df_polars):
        """Test storing and retrieving Polars DataFrame from cache."""
        cache = BarCache(cache_dir=temp_cache_dir)

        cache.put(
            df=sample_bar_df_polars,
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
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_bar_df_polars)
        assert result.shape == sample_bar_df_polars.shape

    def test_get_returns_none_when_not_cached(self, temp_cache_dir):
        """Test that get returns None if data not in cache."""
        cache = BarCache(cache_dir=temp_cache_dir)

        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )

        assert result is None

    def test_get_range_returns_polars(self, temp_cache_dir, sample_bar_df_polars):
        """Test get_range returns concatenated Polars DataFrame."""
        cache = BarCache(cache_dir=temp_cache_dir)

        # Store data for 3 consecutive days
        for day in range(1, 4):
            cache.put(
                df=sample_bar_df_polars,
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
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_bar_df_polars) * 3

    def test_get_range_returns_none_if_any_missing(self, temp_cache_dir, sample_bar_df_polars):
        """Test get_range returns None if any day missing from range."""
        cache = BarCache(cache_dir=temp_cache_dir)

        # Store data for days 1 and 3, but skip day 2
        for day in [1, 3]:
            cache.put(
                df=sample_bar_df_polars,
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

        # Should return None because day 2 is missing
        assert result is None

    def test_put_and_get_preserves_data_types(self, temp_cache_dir):
        """Test that data types are preserved through cache round-trip."""
        cache = BarCache(cache_dir=temp_cache_dir)

        df = pl.DataFrame(
            {
                "symbol": ["BTCUSDT", "ETHUSDT"],
                "volume": [1000.5, 2000.5],
                "count": [10, 20],
            }
        )

        cache.put(
            df=df,
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
        assert result.shape == df.shape
        # Check column names and types match
        assert result.columns == df.columns
        assert result.dtypes == df.dtypes
