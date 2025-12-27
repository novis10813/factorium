"""
Data loader for Binance market data.

Provides synchronous interface for loading Parquet data (Hive partitioned) via DuckDB.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import duckdb
import pandas as pd

from .aggbar import AggBar
from .bar import TimeBar
from .utils.fetch import BinanceDataDownloader
from .utils.parquet import get_market_string, build_hive_path


class BinanceDataLoader:
    """
    Data loader for Binance market data with automatic download.
    
    Uses DuckDB to query Parquet files stored in Hive partition format.
    
    Args:
        base_path: Base directory for data storage
        max_concurrent_downloads: Maximum number of concurrent downloads
        retry_attempts: Number of retry attempts for failed downloads
        retry_delay: Delay between retries in seconds
    
    Example:
        >>> loader = BinanceDataLoader()
        >>> df = loader.load_data(
        ...     symbol="BTCUSDT",
        ...     data_type="trades",
        ...     market_type="futures",
        ...     futures_type="um",
        ...     start_date="2024-01-01",
        ...     days=7
        ... )
    """
    
    def __init__(
        self,
        base_path: str = "./Data",
        max_concurrent_downloads: int = 5,
        retry_attempts: int = 3,
        retry_delay: int = 1
    ):
        self.base_path = Path(base_path)
        self.downloader = BinanceDataDownloader(
            base_path=base_path,
            max_concurrent_downloads=max_concurrent_downloads,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay
        )
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        futures_type: str = 'cm',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        columns: Optional[List[str]] = None,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Load data for specified parameters (synchronous interface).
        
        If local files are missing (or force_download=True), triggers automatic download.
        
        Args:
            symbol: Trading symbol
            data_type: Data type (trades/klines/aggTrades)
            market_type: Market type (spot/futures)
            futures_type: Futures type (cm/um)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days: Number of days to load
            columns: Specific columns to return
            force_download: Force re-download even if files exist
            
        Returns:
            DataFrame with loaded data
        """
        start_dt, end_dt = self._calculate_date_range(start_date, end_date, days)
        resolved_start = start_dt.strftime("%Y-%m-%d")
        resolved_end = end_dt.strftime("%Y-%m-%d")

        if force_download or not self._check_all_files_exist(
            symbol, data_type, market_type, futures_type,
            start_dt, end_dt,
        ):
            asyncio.run(
                self.downloader.download_data(
                    symbol=symbol,
                    data_type=data_type,
                    market_type=market_type,
                    futures_type=futures_type,
                    start_date=resolved_start,
                    end_date=resolved_end,
                )
            )
            self.logger.info("Download completed.")

        return self._read_data_duckdb(
            symbol, data_type, market_type, futures_type,
            start_dt, end_dt, columns=columns,
        )
    
    def _read_data_duckdb(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        futures_type: str,
        start_dt: datetime,
        end_dt: datetime,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read Parquet files using DuckDB with Hive partitioning."""
        market = get_market_string(market_type, futures_type)
        
        # Build glob pattern for Parquet files
        base_pattern = (
            self.base_path
            / f"market={market}"
            / f"data_type={data_type}"
            / f"symbol={symbol}"
            / "**/*.parquet"
        )
        
        # Build column selection
        col_str = ", ".join(columns) if columns else "*"
        
        # Build date filter conditions
        # We need to filter by year, month, day partitions
        date_conditions = self._build_date_filter(start_dt, end_dt)
        
        query = f"""
            SELECT {col_str}
            FROM read_parquet('{base_pattern}', hive_partitioning=true)
            WHERE {date_conditions}
            ORDER BY year, month, day
        """
        
        try:
            result = duckdb.query(query).df()
            if result.empty:
                raise ValueError(f"No data found for {symbol} between {start_dt.date()} and {end_dt.date()}")
            return result
        except Exception as e:
            self.logger.error(f"Error querying data: {e}")
            raise
    
    def _build_date_filter(self, start_dt: datetime, end_dt: datetime) -> str:
        """Build SQL WHERE clause for date filtering on Hive partitions."""
        # Hive partitions are read as VARCHAR, so we need to cast them to INTEGER
        # Use composite condition for efficient filtering
        start_val = start_dt.year * 10000 + start_dt.month * 100 + start_dt.day
        end_val = end_dt.year * 10000 + end_dt.month * 100 + end_dt.day
        
        return f"(CAST(year AS INTEGER) * 10000 + CAST(month AS INTEGER) * 100 + CAST(day AS INTEGER)) >= {start_val} AND (CAST(year AS INTEGER) * 10000 + CAST(month AS INTEGER) * 100 + CAST(day AS INTEGER)) < {end_val}"
    
    def _check_all_files_exist(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        futures_type: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> bool:
        """Check if all required Parquet files exist."""
        market = get_market_string(market_type, futures_type)
        
        current = start_dt
        while current < end_dt:
            hive_path = build_hive_path(
                self.base_path, market, data_type, symbol,
                current.year, current.month, current.day
            )
            parquet_file = hive_path / "data.parquet"
            if not parquet_file.exists():
                return False
            current += timedelta(days=1)
        return True
    
    def load_aggbar(
        self,
        symbols: List[str],
        data_type: str,
        market_type: str,
        futures_type: str = 'cm',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        columns: Optional[List[str]] = None,
        force_download: bool = False,
        **bar_kwargs,
    ) -> AggBar:
        """
        Load data for multiple symbols and return as AggBar.
        
        Args:
            symbols: List of trading symbols
            data_type: Data type (trades/klines/aggTrades)
            market_type: Market type (spot/futures)
            futures_type: Futures type (cm/um)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days: Number of days to load
            columns: Specific columns to return
            force_download: Force re-download even if files exist
            **bar_kwargs: Additional arguments passed to TimeBar constructor
                         (e.g., timestamp_col, price_col, volume_col, interval_ms)
            
        Returns:
            AggBar object containing aggregated bar data for all symbols
            
        Example:
            >>> loader = BinanceDataLoader()
            >>> agg = loader.load_aggbar(
            ...     symbols=["BTCUSDT", "ETHUSDT"],
            ...     data_type="aggTrades",
            ...     market_type="futures",
            ...     futures_type="um",
            ...     start_date="2024-01-01",
            ...     days=7,
            ...     timestamp_col="transact_time",
            ...     price_col="price",
            ...     volume_col="quantity",
            ...     interval_ms=60_000
            ... )
        """
        bars = []
        for symbol in symbols:
            self.logger.info(f"Loading data for {symbol}...")
            df = self.load_data(
                symbol=symbol,
                data_type=data_type,
                market_type=market_type,
                futures_type=futures_type,
                start_date=start_date,
                end_date=end_date,
                days=days,
                columns=columns,
                force_download=force_download,
            )
            bar = TimeBar(df, **bar_kwargs)
            bars.append(bar)
            self.logger.info(f"Created {len(bar)} bars for {symbol}")
        
        agg = AggBar(bars)
        self.logger.info(f"Created AggBar with {len(agg.symbols)} symbols, {len(agg)} total rows")
        return agg
    
    def _calculate_date_range(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        days: Optional[int]
    ) -> tuple[datetime, datetime]:
        """Calculate date range."""
        if start_date and end_date:
            return (
                datetime.strptime(start_date, "%Y-%m-%d"),
                datetime.strptime(end_date, "%Y-%m-%d")
            )
            
        if start_date and not end_date and days:
            return (
                datetime.strptime(start_date, "%Y-%m-%d"),
                datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days)
            )
        
        end = datetime.now()
        if days:
            start = end - timedelta(days=days-1)
        else:
            start = end - timedelta(days=6)
            
        return start, end

