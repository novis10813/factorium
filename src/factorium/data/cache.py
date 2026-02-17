"""Daily cache layer for pre-aggregated bar data."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl

from ..storage import StorageBackend, LocalStorageBackend


class BarCache:
    """Daily cache for pre-aggregated bar data.

    Cache structure:
    cache_dir/
    └── {cache_key}/
        ├── 2024-01-01.parquet
        ├── 2024-01-02.parquet
        └── ...

    Cache key is a hash of (exchange, symbols, interval_ms, data_type, market_type).
    Each day is stored as a separate Parquet file for efficient partial updates.
    """

    def __init__(
        self,
        storage: "StorageBackend | None" = None,
        cache_prefix: str = ".cache",
        *,
        cache_dir: "Path | str | None" = None,  # Deprecated, for backward compatibility
    ):
        """Initialize cache.

        Args:
            storage: StorageBackend instance. If None, creates LocalStorageBackend.
            cache_prefix: Prefix path for cache files within storage.
            cache_dir: DEPRECATED. Use storage parameter instead.
        """
        if cache_dir is not None:
            # Backward compatibility
            import warnings

            warnings.warn(
                "cache_dir is deprecated, use storage parameter instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self.storage = LocalStorageBackend(str(cache_dir))
            self.cache_prefix = ""
            self.cache_dir = Path(cache_dir)  # for backward compatibility
        elif storage is None:
            self.storage = LocalStorageBackend("./Data")
            self.cache_prefix = cache_prefix
        else:
            self.storage = storage
            self.cache_prefix = cache_prefix

        if self.cache_prefix:
            self.storage.makedirs(self.cache_prefix)

    def _build_cache_key(
        self,
        exchange: str,
        symbols: list[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
    ) -> str:
        """Build deterministic cache key from parameters."""
        key_data = {
            "exchange": exchange,
            "symbols": sorted(symbols),
            "interval_ms": interval_ms,
            "data_type": data_type,
            "market_type": market_type,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _get_cache_path(
        self,
        exchange: str,
        symbols: list[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        date: datetime,
    ) -> str:
        """Get cache file path for a specific date."""
        cache_key = self._build_cache_key(exchange, symbols, interval_ms, data_type, market_type)
        date_str = date.strftime("%Y-%m-%d")
        if self.cache_prefix:
            return f"{self.cache_prefix}/{cache_key}/{date_str}.parquet"
        return f"{cache_key}/{date_str}.parquet"

    def get(
        self,
        exchange: str,
        symbols: list[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        date: datetime,
    ) -> pl.DataFrame | None:
        """Get cached data for a single date.

        Returns:
            Polars DataFrame if cached, None otherwise
        """
        cache_path = self._get_cache_path(exchange, symbols, interval_ms, data_type, market_type, date)

        if self.storage.exists(cache_path):
            return self.storage.read_parquet(cache_path)
        return None

    def get_range(
        self,
        exchange: str,
        symbols: list[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame | None:
        """Get cached data for a date range.

        Returns None if any date in the range is missing from cache.
        """
        dfs = []
        current = start_date

        while current < end_date:
            df = self.get(exchange, symbols, interval_ms, data_type, market_type, current)
            if df is None:
                return None
            dfs.append(df)
            current += timedelta(days=1)

        if not dfs:
            return None

        return pl.concat(dfs)

    def put(
        self,
        df: pl.DataFrame,
        exchange: str,
        symbols: list[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        date: datetime,
    ) -> str:
        """Store data in cache for a single date.

        Returns:
            Path to cached file
        """
        cache_path = self._get_cache_path(exchange, symbols, interval_ms, data_type, market_type, date)
        parent_dir = "/".join(cache_path.split("/")[:-1])
        self.storage.makedirs(parent_dir)
        self.storage.write_parquet(df, cache_path)
        return cache_path

    def clear(self) -> int:
        """Clear all cached data.

        Returns:
            Number of files deleted
        """
        count = 0
        pattern = f"{self.cache_prefix}/*/*.parquet" if self.cache_prefix else "*/*.parquet"
        for file_path in self.storage.glob(pattern):
            self.storage.delete(file_path)
            count += 1
        return count
