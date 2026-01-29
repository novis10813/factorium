"""
Data loader for Binance market data.

Provides synchronous interface for loading Parquet data (Hive partitioned) via DuckDB.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Literal

import duckdb
import pandas as pd
import polars as pl


def _run_async(coro):
    """
    Run an async coroutine, handling the case where an event loop is already running.

    This is necessary for Jupyter notebooks which already have a running event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, use asyncio.run()
        return asyncio.run(coro)

    # There's a running loop (e.g., Jupyter), use nest_asyncio if available
    try:
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.run(coro)
    except ImportError:
        # Fall back to creating a new task in the existing loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()


from ..aggbar import AggBar
from .adapters.binance import BinanceAdapter
from .aggregator import BarAggregator
from .cache import BarCache
from .downloader import BinanceDataDownloader
from .metadata import AggBarMetadata
from .parquet import get_market_string, build_hive_path


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
        >>> agg = loader.load_aggbar(
        ...     symbols=["BTCUSDT"],
        ...     data_type="aggTrades",
        ...     market_type="futures",
        ...     futures_type="um",
        ...     start_date="2024-01-01",
        ...     days=7,
        ...     bar_type="time",
        ...     interval=60_000,
        ... )
    """

    def __init__(
        self,
        base_path: str = "./Data",
        max_concurrent_downloads: int = 5,
        retry_attempts: int = 3,
        retry_delay: int = 1,
    ):
        self.base_path = Path(base_path)
        self.downloader = BinanceDataDownloader(
            base_path=base_path,
            max_concurrent_downloads=max_concurrent_downloads,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
        )
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

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
                self.base_path, market, data_type, symbol, current.year, current.month, current.day
            )
            parquet_file = hive_path / "data.parquet"
            if not parquet_file.exists():
                return False
            current += timedelta(days=1)
        return True

    def load_aggbar(
        self,
        symbols: str | List[str],
        data_type: str,
        market_type: str,
        futures_type: str = "um",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        bar_type: Literal["time", "tick", "volume", "dollar"] = "time",
        interval: float = 60_000,
        force_download: bool = False,
        use_cache: bool = True,
    ) -> AggBar:
        """Load aggregated bar data using DuckDB SQL aggregation.

        Unifies bar aggregation across multiple bar types with automatic
        download and optional caching for time bars.

        Args:
            symbols: Single symbol (str) or list of symbols.
                     Note: tick/volume/dollar bars only support single symbol.
            data_type: Data type (trades/aggTrades)
            market_type: Market type (spot/futures)
            futures_type: Futures type (cm/um), default "um"
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days: Number of days to load
            bar_type: Bar type - "time", "tick", "volume", or "dollar". Default "time"
            interval: Bar interval (meaning depends on bar_type):
                      - time: milliseconds (default 60_000 = 1 minute)
                      - tick: number of ticks
                      - volume: volume threshold
                      - dollar: dollar volume threshold
            force_download: Force re-download even if files exist. Default False
            use_cache: Use aggregation cache (only for time bars). Default True

        Returns:
            AggBar object containing aggregated bar data

        Raises:
            ValueError: If bar_type is not "time" and multiple symbols are provided
            ValueError: If no data found for the specified parameters

        Example:
            >>> loader = BinanceDataLoader()
            >>> # Time bars (multiple symbols)
            >>> agg = loader.load_aggbar(
            ...     symbols=["BTCUSDT", "ETHUSDT"],
            ...     data_type="aggTrades",
            ...     market_type="futures",
            ...     futures_type="um",
            ...     start_date="2024-01-01",
            ...     days=7,
            ...     bar_type="time",
            ...     interval=60_000  # 1 minute
            ... )
            >>> # Tick bars (single symbol)
            >>> agg = loader.load_aggbar(
            ...     symbols="BTCUSDT",
            ...     data_type="aggTrades",
            ...     market_type="futures",
            ...     futures_type="um",
            ...     start_date="2024-01-01",
            ...     days=7,
            ...     bar_type="tick",
            ...     interval=1000  # 1000 ticks per bar
            ... )
        """
        # Normalize symbols to list
        if isinstance(symbols, str):
            symbols = [symbols]

        # Check if this is klines data (bypass aggregation)
        is_klines = data_type == "klines"
        if is_klines:
            # Klines doesn't support bar_type parameter - it's already OHLCV
            if bar_type != "time":
                raise ValueError(
                    f"data_type='klines' only supports bar_type='time', got '{bar_type}'. "
                    "Klines data is already aggregated OHLCV."
                )

        # Validate: non-time bars only support single symbol
        if bar_type != "time" and len(symbols) > 1:
            raise ValueError(
                f"bar_type='{bar_type}' only supports single symbol, got {len(symbols)} symbols. "
                "Use bar_type='time' for multi-symbol aggregation."
            )

        # Calculate date range
        start_dt, end_dt = self._calculate_date_range(start_date, end_date, days)

        # Download missing data
        if force_download:
            self._download_all_symbols(symbols, data_type, market_type, futures_type, start_dt, end_dt)
        else:
            missing = self._find_missing_files(symbols, data_type, market_type, futures_type, start_dt, end_dt)
            if missing:
                self._download_missing_files(missing, data_type, market_type, futures_type)

        # ===== KLINES: Direct loading without aggregation =====
        if is_klines:
            return self._load_klines_direct(
                symbols=symbols,
                data_type=data_type,
                market_type=market_type,
                futures_type=futures_type,
                start_dt=start_dt,
                end_dt=end_dt,
                interval_ms=int(interval),
            )

        # Initialize components (for trades/aggTrades aggregation)
        adapter = BinanceAdapter()
        aggregator = BarAggregator()
        cache = BarCache() if (use_cache and bar_type == "time") else None
        market_str = self._get_market_string(market_type, futures_type)

        # Collect aggregated data
        all_dfs: List[pl.DataFrame] = []
        all_metadata: List[AggBarMetadata] = []
        current = start_dt

        # For non-time bars, process entire date range at once (no daily chunking)
        if bar_type != "time":
            start_ts = int(current.timestamp() * 1000)
            end_ts = int(end_dt.timestamp() * 1000)

            parquet_pattern = adapter.build_parquet_glob(
                base_path=self.base_path,
                symbols=symbols,
                data_type=data_type,
                market_type=market_type,
                futures_type=futures_type,
            )

            column_mapping = adapter.get_column_mapping(data_type)
            include_buyer_seller = data_type in ("trades", "aggTrades")

            # Select aggregation method based on bar_type
            if bar_type == "tick":
                df, meta = aggregator.aggregate_tick_bars(
                    parquet_pattern=parquet_pattern,
                    symbol=symbols[0],
                    interval_ticks=int(interval),
                    column_mapping=column_mapping,
                    include_buyer_seller=include_buyer_seller,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            elif bar_type == "volume":
                df, meta = aggregator.aggregate_volume_bars(
                    parquet_pattern=parquet_pattern,
                    symbol=symbols[0],
                    interval_volume=float(interval),
                    column_mapping=column_mapping,
                    include_buyer_seller=include_buyer_seller,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            elif bar_type == "dollar":
                df, meta = aggregator.aggregate_dollar_bars(
                    parquet_pattern=parquet_pattern,
                    symbol=symbols[0],
                    interval_dollar=float(interval),
                    column_mapping=column_mapping,
                    include_buyer_seller=include_buyer_seller,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )

            if not df.is_empty():
                all_dfs.append(df)
                all_metadata.append(meta)
        else:
            # Time bars: process day by day with caching
            while current < end_dt:
                # Try cache first
                if cache:
                    cached_df = cache.get(
                        exchange=adapter.name,
                        symbols=symbols,
                        interval_ms=int(interval),
                        data_type=data_type,
                        market_type=market_str,
                        date=current,
                    )
                    if cached_df is not None:
                        self.logger.debug(f"Cache hit for {current.date()}")
                        # cached_df is already a Polars DataFrame
                        all_dfs.append(cached_df)

                        # Compute metadata for cached data
                        cached_meta = AggBarMetadata(
                            symbols=cached_df["symbol"].unique().sort().to_list(),
                            min_time=cached_df["start_time"].min(),
                            max_time=cached_df["end_time"].max(),
                            num_rows=len(cached_df),
                        )
                        all_metadata.append(cached_meta)

                        current += timedelta(days=1)
                        continue

                # Cache miss: aggregate using DuckDB
                self.logger.debug(f"Cache miss for {current.date()}, aggregating...")
                day_start_ts = int(current.timestamp() * 1000)
                day_end_ts = int((current + timedelta(days=1)).timestamp() * 1000)

                parquet_pattern = adapter.build_parquet_glob(
                    base_path=self.base_path,
                    symbols=symbols,
                    data_type=data_type,
                    market_type=market_type,
                    futures_type=futures_type,
                )

                column_mapping = adapter.get_column_mapping(data_type)
                include_buyer_seller = data_type in ("trades", "aggTrades")

                df, meta = aggregator.aggregate_time_bars(
                    parquet_pattern=parquet_pattern,
                    symbols=symbols,
                    interval_ms=int(interval),
                    start_ts=day_start_ts,
                    end_ts=day_end_ts,
                    column_mapping=column_mapping,
                    include_buyer_seller=include_buyer_seller,
                )

                if not df.is_empty():
                    # Store in cache (df is already Polars)
                    if cache:
                        cache.put(
                            df=df,
                            exchange=adapter.name,
                            symbols=symbols,
                            interval_ms=int(interval),
                            data_type=data_type,
                            market_type=market_str,
                            date=current,
                        )
                    all_dfs.append(df)
                    all_metadata.append(meta)

                current += timedelta(days=1)

        # Validate we have data
        if not all_dfs:
            raise ValueError(f"No data found for {symbols} between {start_dt.date()} and {end_dt.date()}")

        # Combine and return
        result_df = pl.concat(all_dfs)
        combined_meta = (
            AggBarMetadata.merge(all_metadata)
            if all_metadata
            else AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)
        )
        self.logger.info(f"Loaded {len(result_df)} bars for {len(symbols)} symbols")

        return AggBar(result_df, combined_meta)

    def _calculate_date_range(
        self, start_date: Optional[str], end_date: Optional[str], days: Optional[int]
    ) -> tuple[datetime, datetime]:
        """Calculate date range."""
        if start_date and end_date:
            return (datetime.strptime(start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d"))

        if start_date and not end_date and days:
            return (
                datetime.strptime(start_date, "%Y-%m-%d"),
                datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=days),
            )

        end = datetime.now()
        if days:
            start = end - timedelta(days=days - 1)
        else:
            start = end - timedelta(days=6)

        return start, end

    def load_aggbar_fast(
        self,
        symbols: List[str],
        data_type: str,
        market_type: str,
        futures_type: str = "um",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        interval_ms: int = 60_000,
        force_download: bool = False,
        use_cache: bool = True,
    ) -> AggBar:
        """Deprecated: Use load_aggbar instead.

        This method is maintained for backward compatibility only.
        All functionality has been merged into load_aggbar with the bar_type parameter.

        Args:
            symbols: List of trading symbols
            data_type: Data type (trades/aggTrades)
            market_type: Market type (spot/futures)
            futures_type: Futures type (cm/um)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days: Number of days to load
            interval_ms: Bar interval in milliseconds (default: 60000 = 1 minute)
            force_download: Force re-download even if files exist
            use_cache: Use aggregation cache (default: True)

        Returns:
            AggBar object containing aggregated bar data
        """
        import warnings

        warnings.warn(
            "load_aggbar_fast is deprecated, use load_aggbar instead with bar_type='time' and interval=interval_ms",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load_aggbar(
            symbols=symbols,
            data_type=data_type,
            market_type=market_type,
            futures_type=futures_type,
            start_date=start_date,
            end_date=end_date,
            days=days,
            bar_type="time",
            interval=interval_ms,
            force_download=force_download,
            use_cache=use_cache,
        )

    def _get_market_string(self, market_type: str, futures_type: str) -> str:
        """Get market string for cache key."""
        if market_type == "futures":
            return f"futures_{futures_type}"
        return market_type

    def _check_all_symbols_exist(
        self,
        symbols: List[str],
        data_type: str,
        market_type: str,
        futures_type: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> bool:
        """Check if all required Parquet files exist for all symbols."""
        for symbol in symbols:
            if not self._check_all_files_exist(symbol, data_type, market_type, futures_type, start_dt, end_dt):
                return False
        return True

    def _find_missing_files(
        self,
        symbols: List[str],
        data_type: str,
        market_type: str,
        futures_type: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> dict[str, list[datetime]]:
        """Find missing (symbol, date) pairs instead of all-or-nothing check.

        Returns:
            Dict mapping symbol -> list of missing dates
        """
        market = get_market_string(market_type, futures_type)
        missing: dict[str, list[datetime]] = {}

        for symbol in symbols:
            symbol_missing: list[datetime] = []
            current = start_dt
            while current < end_dt:
                hive_path = build_hive_path(
                    self.base_path, market, data_type, symbol, current.year, current.month, current.day
                )
                parquet_file = hive_path / "data.parquet"
                if not parquet_file.exists():
                    symbol_missing.append(current)
                current += timedelta(days=1)

            if symbol_missing:
                missing[symbol] = symbol_missing

        return missing

    def _download_missing_files(
        self,
        missing: dict[str, list[datetime]],
        data_type: str,
        market_type: str,
        futures_type: str,
    ) -> None:
        """Download only missing (symbol, date) pairs.

        Groups consecutive dates to minimize download calls.
        """
        if not missing:
            return

        async def download_all():
            tasks = []
            for symbol, dates in missing.items():
                ranges = self._group_consecutive_dates(dates)
                for range_start, range_end in ranges:
                    tasks.append(
                        self.downloader.download_data(
                            symbol=symbol,
                            data_type=data_type,
                            market_type=market_type,
                            futures_type=futures_type,
                            start_date=range_start.strftime("%Y-%m-%d"),
                            end_date=(range_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                        )
                    )

            total_ranges = len(tasks)
            total_symbols = len(missing)
            self.logger.info(f"Downloading {total_ranges} date ranges for {total_symbols} symbols...")
            await asyncio.gather(*tasks)

        _run_async(download_all())

    def _group_consecutive_dates(self, dates: list[datetime]) -> list[tuple[datetime, datetime]]:
        """Group consecutive dates into (start, end) ranges."""
        if not dates:
            return []

        sorted_dates = sorted(dates)
        ranges: list[tuple[datetime, datetime]] = []
        range_start = sorted_dates[0]
        range_end = sorted_dates[0]

        for date in sorted_dates[1:]:
            if (date - range_end).days == 1:
                range_end = date
            else:
                ranges.append((range_start, range_end))
                range_start = date
                range_end = date

        ranges.append((range_start, range_end))
        return ranges

    def _download_all_symbols(
        self,
        symbols: List[str],
        data_type: str,
        market_type: str,
        futures_type: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> None:
        """Download data for all symbols in parallel."""
        resolved_start = start_dt.strftime("%Y-%m-%d")
        resolved_end = end_dt.strftime("%Y-%m-%d")

        async def download_all():
            tasks = [
                self.downloader.download_data(
                    symbol=symbol,
                    data_type=data_type,
                    market_type=market_type,
                    futures_type=futures_type,
                    start_date=resolved_start,
                    end_date=resolved_end,
                )
                for symbol in symbols
            ]
            self.logger.info(f"Downloading {len(symbols)} symbols in parallel...")
            await asyncio.gather(*tasks)

        _run_async(download_all())

    def _load_klines_direct(
        self,
        symbols: List[str],
        data_type: str,
        market_type: str,
        futures_type: str,
        start_dt: datetime,
        end_dt: datetime,
        interval_ms: int,
    ) -> AggBar:
        """Load klines data directly without aggregation.

        Klines data is already in OHLCV format, so we just:
        1. Load the Parquet files using DuckDB
        2. Rename columns to match AggBar schema
        3. Optionally resample to different intervals

        Args:
            symbols: List of symbols to load
            data_type: Data type (must be "klines")
            market_type: Market type (spot/futures)
            futures_type: Futures type (cm/um)
            start_dt: Start datetime
            end_dt: End datetime
            interval_ms: Target interval in milliseconds (for resampling)

        Returns:
            AggBar object containing klines OHLCV data
        """
        import duckdb

        adapter = BinanceAdapter()
        market_str = self._get_market_string(market_type, futures_type)

        # Build parquet glob pattern
        parquet_pattern = adapter.build_parquet_glob(
            base_path=self.base_path,
            symbols=symbols,
            data_type=data_type,
            market_type=market_type,
            futures_type=futures_type,
        )

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        # Validate timestamp values for SQL safety
        if not isinstance(start_ts, int) or not isinstance(end_ts, int):
            raise ValueError(f"Invalid timestamp values: start_ts={start_ts}, end_ts={end_ts}")
        if start_ts < 0 or end_ts < 0:
            raise ValueError(f"Timestamp values must be non-negative: start_ts={start_ts}, end_ts={end_ts}")
        if start_ts > end_ts:
            raise ValueError(f"start_ts must be <= end_ts: start_ts={start_ts}, end_ts={end_ts}")

        # Load klines data using DuckDB
        # Klines columns: open_time, open, high, low, close, volume, close_time,
        #                 quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore
        # AggBar needs: start_time, end_time, symbol, open, high, low, close, volume,
        #               quote_volume, count, taker_buy_volume, taker_buy_quote_volume

        query = f"""
        SELECT 
            open_time as start_time,
            close_time as end_time,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            count,
            taker_buy_volume,
            taker_buy_quote_volume
        FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
        WHERE open_time >= {start_ts} AND close_time <= {end_ts}
        ORDER BY open_time, symbol
        """

        con = duckdb.connect(":memory:")
        df = con.execute(query).pl()
        con.close()

        if df.is_empty():
            raise ValueError(
                f"No data found for symbols={symbols}, data_type={data_type}, "
                f"market_type={market_str}, date_range={start_dt.date()} to {end_dt.date()}"
            )

        # Resample if needed (interval_ms != 60000 means not 1m)
        # Default klines is 1m (60000ms)
        if interval_ms != 60_000:
            df = self._resample_klines(df, interval_ms)

        self.logger.info(
            f"Loaded {len(df)} klines bars for {len(symbols)} symbols ({start_dt.date()} to {end_dt.date()})"
        )

        return AggBar(df)

    def _resample_klines(self, df: pl.DataFrame, interval_ms: int) -> pl.DataFrame:
        """Resample 1m klines to a different interval.

        Args:
            df: Polars DataFrame with 1m klines data
            interval_ms: Target interval in milliseconds

        Returns:
            Resampled DataFrame
        """
        # Convert interval_ms to Polars duration string
        # Examples: 60000 -> "1m", 300000 -> "5m", 3600000 -> "1h"
        if interval_ms % 86400000 == 0:
            interval_str = f"{interval_ms // 86400000}d"
        elif interval_ms % 3600000 == 0:
            interval_str = f"{interval_ms // 3600000}h"
        elif interval_ms % 60000 == 0:
            interval_str = f"{interval_ms // 60000}m"
        else:
            raise ValueError(
                f"interval_ms={interval_ms} cannot be converted to standard time unit. Use multiples of 1m, 1h, or 1d."
            )

        # Group by symbol and time bucket for proper resampling
        # For OHLCV: open=first, high=max, low=min, close=last, volume=sum
        # For microstructure data: all sum
        resampled = (
            df.with_columns(
                [
                    pl.from_epoch("start_time", time_unit="ms").alias("start_dt"),
                    pl.from_epoch("start_time", time_unit="ms").dt.truncate(interval_str).alias("time_bucket"),
                ]
            )
            .sort(["symbol", "start_dt"])
            .group_by(["symbol", "time_bucket"])
            .agg(
                [
                    pl.col("time_bucket").first().alias("start_dt"),
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                    pl.col("quote_volume").sum().alias("quote_volume"),
                    pl.col("count").sum().alias("count"),
                    pl.col("taker_buy_volume").sum().alias("taker_buy_volume"),
                    pl.col("taker_buy_quote_volume").sum().alias("taker_buy_quote_volume"),
                ]
            )
            .with_columns(
                [
                    pl.col("start_dt").dt.epoch("ms").alias("start_time"),
                    (pl.col("start_dt").dt.epoch("ms") + interval_ms - 1).alias("end_time"),
                ]
            )
            .select(
                [
                    "start_time",
                    "end_time",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                    "count",
                    "taker_buy_volume",
                    "taker_buy_quote_volume",
                ]
            )
            .sort(["start_time", "symbol"])
        )

        return resampled
