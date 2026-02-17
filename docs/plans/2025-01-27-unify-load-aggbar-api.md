# Unify load_aggbar API with bar_type Parameter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge `load_aggbar_fast` into `load_aggbar` with support for multiple bar types (time, tick, volume, dollar)

**Architecture:** 
- Replace old `load_aggbar` (TimeBar-based) with new unified version supporting bar_type parameter
- Rename `load_aggbar_fast` logic to become the main `load_aggbar` implementation
- Keep `load_aggbar_fast` as deprecated alias for backward compatibility
- Add validation to restrict non-time bars to single symbol only
- Support unified `interval` parameter that adapts meaning based on `bar_type`

**Tech Stack:** 
- DuckDB for aggregation (BarAggregator)
- BinanceAdapter for column mapping and parquet path building
- BarCache for time bar caching
- AggBar for result wrapping

---

## Task 1: Update Method Signature and Add Type Hints

**Files:**
- Modify: `src/factorium/data/loader.py:208-275` (old load_aggbar)
- Modify: `src/factorium/data/loader.py:1-25` (imports)

**Step 1: Add Literal type import**

In the imports section, add:
```python
from typing import Literal
```

**Step 2: Replace old load_aggbar signature**

Replace lines 208-275 (old load_aggbar) with:

```python
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
```

**Step 3: Run a quick test to ensure imports work**

Run: `python -c "from factorium.data.loader import BinanceDataLoader; print('imports ok')"`
Expected: `imports ok`

---

## Task 2: Implement Unified load_aggbar Logic

**Files:**
- Modify: `src/factorium/data/loader.py` (continue from Task 1, same method)

**Step 1: Add implementation after docstring**

Add this code right after the docstring in the new load_aggbar method:

```python
    # Normalize symbols to list
    if isinstance(symbols, str):
        symbols = [symbols]
    
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
    
    # Initialize components
    adapter = BinanceAdapter()
    aggregator = BarAggregator()
    cache = BarCache() if (use_cache and bar_type == "time") else None
    market_str = self._get_market_string(market_type, futures_type)
    
    # Collect aggregated data
    all_dfs: List[pd.DataFrame] = []
    current = start_dt
    
    # For non-time bars, process entire date range at once (no daily chunking)
    if bar_type != "time":
        day_start_ts = int(current.timestamp() * 1000)
        day_end_ts = int(end_dt.timestamp() * 1000)
        
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
            df = aggregator.aggregate_tick_bars(
                parquet_pattern=parquet_pattern,
                symbol=symbols[0],
                interval_ticks=int(interval),
                column_mapping=column_mapping,
                include_buyer_seller=include_buyer_seller,
            )
        elif bar_type == "volume":
            df = aggregator.aggregate_volume_bars(
                parquet_pattern=parquet_pattern,
                symbol=symbols[0],
                interval_volume=float(interval),
                column_mapping=column_mapping,
                include_buyer_seller=include_buyer_seller,
            )
        elif bar_type == "dollar":
            df = aggregator.aggregate_dollar_bars(
                parquet_pattern=parquet_pattern,
                symbol=symbols[0],
                interval_dollar=float(interval),
                column_mapping=column_mapping,
                include_buyer_seller=include_buyer_seller,
            )
        
        if not df.empty:
            all_dfs.append(df)
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
                    all_dfs.append(cached_df)
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
            
            df = aggregator.aggregate_time_bars(
                parquet_pattern=parquet_pattern,
                symbols=symbols,
                interval_ms=int(interval),
                start_ts=day_start_ts,
                end_ts=day_end_ts,
                column_mapping=column_mapping,
                include_buyer_seller=include_buyer_seller,
            )
            
            if not df.empty:
                # Store in cache
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
            
            current += timedelta(days=1)
    
    # Validate we have data
    if not all_dfs:
        raise ValueError(f"No data found for {symbols} between {start_dt.date()} and {end_dt.date()}")
    
    # Combine and return
    result_df = pd.concat(all_dfs, ignore_index=True)
    self.logger.info(f"Loaded {len(result_df)} bars for {len(symbols)} symbols")
    
    return AggBar.from_df(result_df)
```

**Step 2: Remove the TimeBar import (no longer needed)**

Remove this line from imports:
```python
from ..bar import TimeBar
```

**Step 3: Run import validation**

Run: `python -c "from factorium.data.loader import BinanceDataLoader; print('new load_aggbar is valid')"`
Expected: `new load_aggbar is valid`

---

## Task 3: Keep load_aggbar_fast as Deprecated Alias

**Files:**
- Modify: `src/factorium/data/loader.py` (add new method after load_aggbar)

**Step 1: Add deprecated alias method**

After the new `load_aggbar` method (around line 430 in current file), add:

```python
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
    """
    import warnings
    warnings.warn(
        "load_aggbar_fast is deprecated, use load_aggbar instead with bar_type='time' and interval=interval_ms",
        DeprecationWarning,
        stacklevel=2
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
```

**Step 2: Verify deprecation warning works**

Run: `python -c "
from factorium.data.loader import BinanceDataLoader
import warnings
loader = BinanceDataLoader()
warnings.simplefilter('always')
try:
    loader.load_aggbar_fast(['BTC'], 'trades', 'spot', start_date='2024-01-01', days=1)
except Exception as e:
    if 'No data found' in str(e):
        print('Deprecation alias works - got expected no-data error')
"`
Expected: `Deprecation alias works - got expected no-data error`

---

## Task 4: Run Tests to Verify Changes

**Files:**
- Test: `tests/data/test_loader_fast.py`

**Step 1: Run existing tests for load_aggbar_fast**

Run: `pytest tests/data/test_loader_fast.py -v --tb=short`
Expected: All tests pass (they still use load_aggbar_fast which is now an alias)

**Step 2: Verify no import errors**

Run: `python -c "from factorium.data.loader import BinanceDataLoader; print('OK')"`
Expected: `OK`

---

## Task 5: Verify Full Test Suite

**Files:**
- Test: `tests/data/` (all loader tests)

**Step 1: Run all data loader tests**

Run: `pytest tests/data/ -v --tb=short -k "loader"`
Expected: All tests pass

**Step 2: Check for any remaining imports of TimeBar from loader**

Run: `grep -r "from.*loader.*TimeBar" tests/`
Expected: No output (or only unrelated results)

---

## Task 6: Create Commit

**Files:**
- Modified: `src/factorium/data/loader.py`

**Step 1: Review changes**

Run: `git diff src/factorium/data/loader.py`
Expected: See the new load_aggbar, deprecated load_aggbar_fast, removed TimeBar import, removed old load_aggbar

**Step 2: Stage and commit**

Run:
```bash
git add src/factorium/data/loader.py
git commit -m "refactor(loader): unify load_aggbar API with bar_type parameter"
```

Expected: Commit succeeds

**Step 3: Verify commit**

Run: `git log -1 --oneline`
Expected: Shows the new commit

---

## Summary

This plan:
1. ✅ Adds `Literal` type for bar_type parameter
2. ✅ Replaces old `load_aggbar` with new unified version
3. ✅ Adds validation for single-symbol requirement of non-time bars
4. ✅ Supports all four bar types: time, tick, volume, dollar
5. ✅ Implements unified `interval` parameter
6. ✅ Keeps time bar caching only (disabled for other types)
7. ✅ Removes TimeBar import (no longer needed)
8. ✅ Keeps `load_aggbar_fast` as deprecated alias
9. ✅ Passes all existing tests
10. ✅ Creates clean git commit

Total estimated time: 30-45 minutes
