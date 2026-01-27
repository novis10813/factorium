"""Daily cache layer for pre-aggregated bar data."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


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

    def __init__(self, cache_dir: Path = Path("./Data/.cache")):
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
    ) -> Path:
        """Get cache file path for a specific date."""
        cache_key = self._build_cache_key(exchange, symbols, interval_ms, data_type, market_type)
        date_str = date.strftime("%Y-%m-%d")
        return self.cache_dir / cache_key / f"{date_str}.parquet"

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

        if cache_path.exists():
            return pl.read_parquet(cache_path)
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
    ) -> Path:
        """Store data in cache for a single date.

        Returns:
            Path to cached file
        """
        cache_path = self._get_cache_path(exchange, symbols, interval_ms, data_type, market_type, date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
        return cache_path

    def clear(self) -> int:
        """Clear all cached data.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_subdir in self.cache_dir.iterdir():
            if cache_subdir.is_dir():
                for f in cache_subdir.glob("*.parquet"):
                    f.unlink()
                    count += 1
                cache_subdir.rmdir()
        return count
