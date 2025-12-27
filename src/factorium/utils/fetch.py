"""
Binance market data downloader.

Provides async download functionality for Binance Vision historical data.
"""

import asyncio
import aiohttp
import aiofiles
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional, Tuple, List
import hashlib
import zipfile
import argparse
import tempfile

from .parquet import csv_to_parquet, build_hive_path, get_market_string


class BinanceDataDownloader:
    """
    Asynchronous downloader for Binance Vision historical data.
    
    Args:
        base_path: Base directory for data storage
        max_concurrent_downloads: Maximum number of concurrent downloads
        retry_attempts: Number of retry attempts for failed downloads
        retry_delay: Delay between retries in seconds
    
    Example:
        >>> downloader = BinanceDataDownloader(base_path="./Data")
        >>> await downloader.download_data(
        ...     symbol="BTCUSDT",
        ...     data_type="trades",
        ...     market_type="futures",
        ...     futures_type="um",
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31"
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
        self.max_concurrent_downloads = max_concurrent_downloads
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    async def download_data(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        futures_type: str = 'cm',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None
    ) -> None:
        """
        Download data for specified parameters.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT, BTCUSD_PERP)
            data_type: Data type (trades/klines/aggTrades)
            market_type: Market type (spot/futures)
            futures_type: Futures type (cm/um)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days: Number of days to download
        """
        self._validate_params(data_type, market_type, futures_type)
        start_date_dt, end_date_dt = self._calculate_date_range(start_date, end_date, days)
        download_dir = self._setup_download_dir(symbol, data_type, market_type, futures_type)
        dates = self._generate_date_list(start_date_dt, end_date_dt)
        
        tasks = []
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        async with aiohttp.ClientSession() as session:
            for date in dates:
                task = asyncio.create_task(
                    self._download_single_day(
                        session,
                        symbol,
                        data_type,
                        market_type,
                        futures_type,
                        date,
                        download_dir,
                        semaphore
                    )
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
        self._update_readme()
        
    async def _download_single_day(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        data_type: str,
        market_type: str,
        futures_type: str,
        date: datetime,
        download_dir: Path,
        semaphore: asyncio.Semaphore
    ) -> None:
        """Download single day data and convert to Parquet with Hive partitioning."""
        async with semaphore:
            date_str = date.strftime("%Y-%m-%d")
            self.logger.info(f"Processing data for {date_str}")
            
            filename = self._build_filename(symbol, data_type, date_str)
            checksum_filename = f"{filename}.CHECKSUM"
            base_url = self._build_base_url(market_type, data_type, symbol, futures_type)
            
            for attempt in range(self.retry_attempts):
                try:
                    # Use temp directory for download
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        data_file_path = temp_path / filename
                        
                        if await self._download_file(session, f"{base_url}/{filename}", data_file_path):
                            checksum_file_path = temp_path / checksum_filename
                            if await self._download_file(session, f"{base_url}/{checksum_filename}", checksum_file_path):
                                if await self._verify_checksum(data_file_path, checksum_file_path):
                                    # Extract ZIP to temp directory
                                    await self._extract_zip(data_file_path)
                                    
                                    # Find extracted CSV file
                                    csv_filename = filename.replace('.zip', '.csv')
                                    csv_path = temp_path / csv_filename
                                    
                                    if csv_path.exists():
                                        # Build Hive partition path and convert to Parquet
                                        market = get_market_string(market_type, futures_type)
                                        hive_path = build_hive_path(
                                            self.base_path, market, data_type, symbol,
                                            date.year, date.month, date.day
                                        )
                                        csv_to_parquet(csv_path, hive_path, data_type=data_type)
                                        self.logger.info(f"✓ Successfully processed {filename} -> {hive_path}")
                                        return
                                    else:
                                        self.logger.error(f"CSV file not found after extraction: {csv_path}")
                    
                    self.logger.warning(f"Attempt {attempt + 1} failed for {date_str}")
                    await asyncio.sleep(self.retry_delay)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {date_str}: {str(e)}")
                    await asyncio.sleep(self.retry_delay)
            
            self.logger.error(f"Failed to download data for {date_str} after {self.retry_attempts} attempts")
    
    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        file_path: Path
    ) -> bool:
        """Download a single file."""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    async with aiofiles.open(file_path, 'wb') as f:
                        await f.write(await response.read())
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {str(e)}")
            return False
    
    async def _verify_checksum(self, data_file: Path, checksum_file: Path) -> bool:
        """Verify file checksum."""
        try:
            async with aiofiles.open(checksum_file, 'r') as f:
                expected_checksum = (await f.read()).split()[0]
            
            async with aiofiles.open(data_file, 'rb') as f:
                file_content = await f.read()
                actual_checksum = hashlib.sha256(file_content).hexdigest()
            
            return expected_checksum == actual_checksum
        except Exception as e:
            self.logger.error(f"Error verifying checksum: {str(e)}")
            return False
    
    async def _extract_zip(self, zip_file: Path) -> None:
        """Extract ZIP file."""
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(zip_file.parent)
        except Exception as e:
            self.logger.error(f"Error extracting {zip_file}: {str(e)}")
    
    def _validate_params(self, data_type: str, market_type: str, futures_type: str) -> None:
        """Validate parameters."""
        if data_type not in ['trades', 'klines', 'aggTrades', 'bookTicker', 'bookDepth']:
            raise ValueError("Invalid data type")
        if market_type not in ['spot', 'futures']:
            raise ValueError("Invalid market type")
        if market_type == 'futures' and futures_type not in ['cm', 'um']:
            raise ValueError("Invalid futures type")
    
    def _calculate_date_range(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        days: Optional[int]
    ) -> Tuple[datetime, datetime]:
        """Calculate date range."""
        try:
            if start_date and end_date:
                try:
                    start = datetime.strptime(start_date, "%Y-%m-%d")
                    end = datetime.strptime(end_date, "%Y-%m-%d")
                    
                    if start > end:
                        raise ValueError("Start date must be earlier than or equal to end date")
                        
                    return start, end
                except ValueError as e:
                    self.logger.error(f"Invalid date format: {str(e)}")
                    raise
            
            end = datetime.now()
            if days:
                if days < 1:
                    raise ValueError("Days must be greater than 0")
                start = end - timedelta(days=days-1)
            else:
                start = end - timedelta(days=6)
                
            return start, end
            
        except Exception as e:
            self.logger.error(f"Error calculating date range: {str(e)}")
            raise
    
    def _setup_download_dir(self, symbol: str, data_type: str, market_type: str, futures_type: str = 'cm') -> Path:
        """Setup download directory (kept for backward compatibility, now uses temp dir)."""
        # Note: With Hive partitioning, we use temp directories for download
        # and build the final path using build_hive_path during conversion
        temp_dir = self.base_path / ".temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    def _generate_date_list(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate list of dates."""
        dates = []
        current = start_date
        while current < end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates
    
    def _build_filename(self, symbol: str, data_type: str, date_str: str) -> str:
        """Build filename."""
        if data_type == "klines":
            return f"{symbol}-1m-{date_str}.zip"
        return f"{symbol}-{data_type}-{date_str}.zip"
    
    def _build_base_url(self, market_type: str, data_type: str, symbol: str, futures_type: str = 'cm') -> str:
        """Build base URL."""
        if market_type == "futures":
            market_path = f"futures/{futures_type}"
        else:
            market_path = "spot"
        
        base_url = f"https://data.binance.vision/data/{market_path}/daily/{data_type}/{symbol}"
        
        # Klines have an additional interval subdirectory
        if data_type == "klines":
            base_url = f"{base_url}/1m"
        
        return base_url
    
    def _update_readme(self) -> None:
        """Update README file."""
        readme_content = f"""# Binance Market Data

This directory contains downloaded market data from Binance Vision.

## Directory Structure

```
data/
├── futures/
│   ├── cm/
│   │   ├── trades/
│   │   ├── klines/
│   │   └── aggTrades/
│   └── um/
│       ├── trades/
│       ├── klines/
│       └── aggTrades/
└── spot/
    ├── trades/
    ├── klines/
    └── aggTrades/
```

## Data Types
- trades: Individual trades
- klines: Candlestick data
- aggTrades: Aggregated trades

## Market Types
- futures: Futures market data (CM/UM)
- spot: Spot market data

Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        with open(self.base_path / "README.md", "w") as f:
            f.write(readme_content)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download Binance market data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 7 days of futures trades data (CM)
  python -m factorium.utils.fetch -s BTCUSD_PERP -t trades -m futures -f cm -d 7
  
  # Download specific date range of futures trades data (UM)
  python -m factorium.utils.fetch -s BTCUSDT -t trades -m futures -f um -r 2024-01-01:2024-01-31
  
  # Download spot market data
  python -m factorium.utils.fetch -s BTCUSDT -t klines -m spot -r 2024-01-01:2024-01-31
        """
    )
    
    parser.add_argument('-s', '--symbol', default='BTCUSD_PERP', help='Trading symbol')
    parser.add_argument('-t', '--data-type', default='trades', 
                        choices=['trades', 'klines', 'aggTrades', 'bookTicker', 'bookDepth'])
    parser.add_argument('-m', '--market-type', default='futures', choices=['spot', 'futures'])
    parser.add_argument('-f', '--futures-type', default='cm', choices=['cm', 'um'])
    parser.add_argument('-d', '--days', type=int, default=7)
    parser.add_argument('-p', '--path', default='./Data')
    parser.add_argument('-r', '--date-range', help='Date range YYYY-MM-DD:YYYY-MM-DD')
    parser.add_argument('--max-concurrent', type=int, default=5)
    parser.add_argument('--retry-attempts', type=int, default=3)
    parser.add_argument('--retry-delay', type=int, default=1)
    
    return parser.parse_args()


async def main():
    args = parse_args()
    
    try:
        downloader = BinanceDataDownloader(
            base_path=args.path,
            max_concurrent_downloads=args.max_concurrent,
            retry_attempts=args.retry_attempts,
            retry_delay=args.retry_delay
        )
        
        start_date = None
        end_date = None
        if args.date_range:
            try:
                start_date, end_date = args.date_range.split(':')
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                print("Error: Invalid date range format. Use YYYY-MM-DD:YYYY-MM-DD")
                return
        
        await downloader.download_data(
            symbol=args.symbol,
            data_type=args.data_type,
            market_type=args.market_type,
            futures_type=args.futures_type,
            start_date=start_date,
            end_date=end_date,
            days=args.days if not args.date_range else None
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return


if __name__ == "__main__":
    asyncio.run(main())
