"""
Data loading and processing module for Binance market data.

Provides:
- BinanceDataLoader: Synchronous interface for loading Parquet data via DuckDB
- BinanceDataDownloader: Async downloader for Binance Vision historical data
- BarAggregator: High-performance DuckDB SQL aggregation
- BarCache: Daily cache layer for pre-aggregated data
- Parquet utilities for Hive partitioning
"""

from .aggregator import BarAggregator
from .cache import BarCache
from .downloader import BinanceDataDownloader
from .loader import BinanceDataLoader
from .metadata import AggBarMetadata
from .parquet import (
    BINANCE_COLUMNS,
    build_hive_path,
    csv_to_parquet,
    get_market_string,
    read_hive_parquet,
)

__all__ = [
    "BinanceDataLoader",
    "BinanceDataDownloader",
    "BarAggregator",
    "BarCache",
    "AggBarMetadata",
    "csv_to_parquet",
    "read_hive_parquet",
    "build_hive_path",
    "get_market_string",
    "BINANCE_COLUMNS",
]
