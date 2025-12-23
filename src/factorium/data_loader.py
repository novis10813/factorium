"""
Data loader for Binance market data.

Provides synchronous interface for loading data with automatic download support.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
import logging

from .utils.fetch import BinanceDataDownloader


class BinanceDataLoader:
    """
    Data loader for Binance market data with automatic download.
    
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
        import asyncio

        start_dt, end_dt = self._calculate_date_range(start_date, end_date, days)
        resolved_start = start_dt.strftime("%Y-%m-%d")
        resolved_end = end_dt.strftime("%Y-%m-%d")

        if force_download or not self._check_all_files_exist(
            symbol, data_type, market_type, futures_type,
            resolved_start, resolved_end, days=None,
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

        return self._read_data(
            symbol, data_type, market_type, futures_type,
            resolved_start, resolved_end, days=None, columns=columns,
        )
    
    def _read_data(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        futures_type: str = 'cm',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Read data files (no download)."""
        start, end = self._calculate_date_range(start_date, end_date, days)
        data_dir = self._get_data_dir(symbol, data_type, market_type, futures_type)
        
        dfs = []
        current = start
        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            filename = self._build_filename(symbol, data_type, date_str) + ".csv"
            file_path = data_dir / filename
            
            if file_path.exists():
                try:
                    if data_type == "trades":
                        df = self._read_trades_file(file_path, market_type, columns)
                    elif data_type == "klines":
                        df = self._read_klines_file(file_path, columns)
                    else:
                        df = self._read_agg_trades_file(file_path, columns)
                    
                    dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error reading file {filename}: {str(e)}")
            else:
                self.logger.warning(f"File missing: {file_path}")
            
            current += timedelta(days=1)
        
        if not dfs:
            raise ValueError("No data files found for the specified date range")
        
        final_df = pd.concat(dfs, ignore_index=True)
        final_df['symbol'] = symbol
        
        return final_df
    
    def _get_data_dir(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        futures_type: str,
    ) -> Path:
        """Build data directory path."""
        if market_type == "futures":
            return self.base_path / market_type / futures_type / data_type / symbol
        return self.base_path / market_type / data_type / symbol
    
    def _build_filename(
        self,
        symbol: str,
        data_type: str,
        date_str: str,
    ) -> str:
        """Build base filename (without extension)."""
        if data_type == "klines":
            return f"{symbol}-{data_type}-1m-{date_str}"
        return f"{symbol}-{data_type}-{date_str}"
    
    def _check_all_files_exist(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        futures_type: str,
        start_date: Optional[str],
        end_date: Optional[str],
        days: Optional[int],
    ) -> bool:
        """Check if all required CSV files exist."""
        start, end = self._calculate_date_range(start_date, end_date, days)
        data_dir = self._get_data_dir(symbol, data_type, market_type, futures_type)

        current = start
        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            filename = self._build_filename(symbol, data_type, date_str) + ".csv"
            file_path = data_dir / filename
            if not file_path.exists():
                return False
            current += timedelta(days=1)
        return True
    
    def _read_trades_file(self, file_path: Path, market_type: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Read trades data file."""
        if market_type == "spot":
            df = pd.read_csv(file_path, names=["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "ignore"])
            df = df.drop(columns=["ignore"])
        else:
            df = pd.read_csv(file_path)
        if columns:
            df = df[columns]
        return df
    
    def _read_klines_file(self, file_path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Read klines data file."""
        df = pd.read_csv(file_path)
        if columns:
            df = df[columns]
        return df
    
    def _read_agg_trades_file(self, file_path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Read aggregated trades data file."""
        df = pd.read_csv(file_path)
        if columns:
            df = df[columns]
        return df
    
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
