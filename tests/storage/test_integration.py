"""Integration tests for storage backends with data loading."""

import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from factorium.storage import get_storage_backend, LocalStorageBackend
from factorium.data.cache import BarCache


class TestStorageIntegration:
    """Integration tests for storage with cache."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cache_with_local_backend(self, temp_dir):
        """BarCache should work with LocalStorageBackend."""
        backend = LocalStorageBackend(str(temp_dir))
        cache = BarCache(storage=backend, cache_prefix=".cache")

        # Create test data
        df = pl.DataFrame(
            {
                "start_time": [1704067200000],
                "symbol": ["BTCUSDT"],
                "open": [42000.0],
                "high": [42500.0],
                "low": [41800.0],
                "close": [42200.0],
                "volume": [100.0],
            }
        )

        # Store in cache
        cache.put(
            df=df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60000,
            data_type="aggTrades",
            market_type="futures_um",
            date=datetime(2024, 1, 1),
        )

        # Retrieve from cache
        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60000,
            data_type="aggTrades",
            market_type="futures_um",
            date=datetime(2024, 1, 1),
        )

        assert result is not None
        assert len(result) == 1
        assert result["symbol"][0] == "BTCUSDT"

    def test_factory_creates_correct_backend(self, temp_dir):
        """get_storage_backend should create correct backend type."""
        local_backend = get_storage_backend("local", str(temp_dir))
        assert isinstance(local_backend, LocalStorageBackend)

    def test_local_backend_full_path(self, temp_dir):
        """LocalStorageBackend.full_path should return absolute path."""
        backend = LocalStorageBackend(str(temp_dir))
        full_path = backend.full_path("test/file.parquet")

        assert str(temp_dir) in full_path
        assert "test/file.parquet" in full_path
        assert Path(full_path).is_absolute()

    def test_storage_round_trip(self, temp_dir):
        """Data should survive write/read cycle through storage."""
        backend = LocalStorageBackend(str(temp_dir))

        original_df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
                "c": [1.1, 2.2, 3.3],
            }
        )

        backend.write_parquet(original_df, "test.parquet")
        loaded_df = backend.read_parquet("test.parquet")

        assert original_df.equals(loaded_df)
