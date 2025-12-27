"""
Data loading and processing module for Binance market data.

Provides:
- BinanceDataLoader: Synchronous interface for loading Parquet data via DuckDB
- BinanceDataDownloader: Async downloader for Binance Vision historical data
- Parquet utilities for Hive partitioning
"""

from .loader import BinanceDataLoader
from .downloader import BinanceDataDownloader
from .parquet import (
    csv_to_parquet,
    read_hive_parquet,
    build_hive_path,
    get_market_string,
    BINANCE_COLUMNS,
)

__all__ = [
    "BinanceDataLoader",
    "BinanceDataDownloader",
    "csv_to_parquet",
    "read_hive_parquet",
    "build_hive_path",
    "get_market_string",
    "BINANCE_COLUMNS",
]
