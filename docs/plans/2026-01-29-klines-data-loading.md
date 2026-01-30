# Klines Data Loading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 支援 klines 資料的載入，繞過 aggregation 邏輯，直接載入 OHLCV 資料並支援 resample

**Architecture:** 
- 修改 `BinanceDataLoader.load_aggbar()` 以檢測 `data_type="klines"`
- 當 `data_type="klines"` 時，直接載入 Parquet 檔案並重新命名欄位，繞過 BarAggregator
- 支援 resample 到其他時間週期（5m, 1h, 1d 等）

**Tech Stack:** Polars, DuckDB, pytest

**Context:**
- 目前 downloader 已經硬編碼使用 `1m` interval (downloader.py:280-295)
- Hive 路徑結構：`market={market}/data_type=klines/symbol={symbol}/year/month/day`
- Klines 資料已經是 OHLCV 格式，無需 aggregation
- 只下載 1m 資料，用 Polars resample 到其他週期

---

### Task 1: 添加 klines 直接載入的測試

**Files:**
- Test: `tests/data/test_loader_klines.py`

**Step 1: 寫一個失敗的測試 - klines 基本載入**

創建新測試檔案 `tests/data/test_loader_klines.py`：

```python
"""Tests for BinanceDataLoader.load_aggbar with klines data."""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import patch

from factorium.data.loader import BinanceDataLoader
from factorium.aggbar import AggBar


@pytest.fixture
def sample_klines_data():
    """Create temporary directory with Hive-partitioned klines Parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for day in [1, 2, 3]:
                partition_path = (
                    tmpdir
                    / "market=futures_um"
                    / "data_type=klines"
                    / f"symbol={symbol}"
                    / "year=2024"
                    / "month=01"
                    / f"day={day:02d}"
                )
                partition_path.mkdir(parents=True, exist_ok=True)

                # Klines data: 1440 bars per day (1 minute bars)
                n_bars = 1440
                base_ts = int(datetime(2024, 1, day).timestamp() * 1000)

                df = pd.DataFrame(
                    {
                        "open_time": base_ts + np.arange(n_bars) * 60000,
                        "open": 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1),
                        "high": 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1) + 0.5,
                        "low": 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1) - 0.5,
                        "close": 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1),
                        "volume": np.abs(np.random.randn(n_bars)) * 100 + 10,
                        "close_time": base_ts + np.arange(n_bars) * 60000 + 59999,
                        "quote_volume": np.abs(np.random.randn(n_bars)) * 1000,
                        "count": np.random.randint(10, 100, n_bars),
                        "taker_buy_volume": np.abs(np.random.randn(n_bars)) * 50,
                        "taker_buy_quote_volume": np.abs(np.random.randn(n_bars)) * 500,
                        "ignore": np.zeros(n_bars),
                    }
                )

                table = pa.Table.from_pandas(df)
                pq.write_table(table, partition_path / "data.parquet")

        yield tmpdir


class TestLoadKlines:
    """Tests for BinanceDataLoader.load_aggbar with klines."""

    def test_returns_aggbar_for_klines(self, sample_klines_data):
        """Test that load_aggbar returns AggBar instance for klines."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=3,
                use_cache=False,
            )

        assert isinstance(result, AggBar)
        assert "BTCUSDT" in result.symbols

    def test_klines_has_all_columns(self, sample_klines_data):
        """Test that klines data has all expected columns."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result = loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                use_cache=False,
            )

        # All klines columns including microstructure data
        required_cols = {
            "symbol", "start_time", "end_time", 
            "open", "high", "low", "close", "volume",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"
        }
        assert required_cols.issubset(set(result.cols))

    def test_klines_bypasses_aggregation(self, sample_klines_data):
        """Test that klines loading bypasses BarAggregator."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch("factorium.data.loader.BarAggregator") as MockAggregator:
            mock_agg_instance = MockAggregator.return_value

            with patch.object(loader, "_check_all_symbols_exist", return_value=True):
                result = loader.load_aggbar(
                    symbols=["BTCUSDT"],
                    data_type="klines",
                    market_type="futures",
                    futures_type="um",
                    start_date="2024-01-01",
                    days=1,
                    use_cache=False,
                )

            # Aggregator should not be called for klines
            assert not mock_agg_instance.aggregate_time_bars.called
            assert not mock_agg_instance.aggregate_tick_bars.called
            assert not mock_agg_instance.aggregate_volume_bars.called
            assert not mock_agg_instance.aggregate_dollar_bars.called

    def test_klines_loads_multiple_symbols(self, sample_klines_data):
        """Test loading klines for multiple symbols."""
        loader = BinanceDataLoader(base_path=sample_klines_data)

        with patch.object(loader, "_check_all_symbols_exist", return_value=True):
            result = loader.load_aggbar(
                symbols=["BTCUSDT", "ETHUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=3,
                use_cache=False,
            )

        assert isinstance(result, AggBar)
        assert set(result.symbols) == {"BTCUSDT", "ETHUSDT"}
```

**Step 2: 執行測試確認它失敗**

執行：`pytest tests/data/test_loader_klines.py::TestLoadKlines::test_returns_aggbar_for_klines -v`

預期：FAIL（klines 目前沒有實作）

**Step 3: 提交測試**

```bash
git add tests/data/test_loader_klines.py
git commit -m "test: add klines data loading tests"
```

---

### Task 2: 實作 klines 直接載入邏輯

**Files:**
- Modify: `src/factorium/data/loader.py`

**Step 1: 在 load_aggbar 中添加 klines 檢測邏輯**

在 `load_aggbar` 方法的 line 193 之後（normalize symbols 之後）添加 klines 檢測：

```python
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
```

**Step 2: 添加 klines 直接載入分支**

在 line 215（初始化 components 之前）添加 klines 處理分支：

```python
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
```

**Step 3: 實作 _load_klines_direct 方法**

在 `BinanceDataLoader` 類別最後添加新方法：

```python
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
        f"Loaded {len(df)} klines bars for {len(symbols)} symbols "
        f"({start_dt.date()} to {end_dt.date()})"
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
            f"interval_ms={interval_ms} cannot be converted to standard time unit. "
            "Use multiples of 1m, 1h, or 1d."
        )
    
    # Group by symbol and resample
    # For OHLCV: open=first, high=max, low=min, close=last, volume=sum
    # For microstructure data: all sum
    resampled = (
        df
        .with_columns([
            pl.from_epoch("start_time", time_unit="ms").alias("start_dt"),
        ])
        .sort(["symbol", "start_dt"])
        .group_by("symbol")
        .agg([
            pl.col("start_dt").dt.truncate(interval_str).alias("start_dt"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("quote_volume").sum().alias("quote_volume"),
            pl.col("count").sum().alias("count"),
            pl.col("taker_buy_volume").sum().alias("taker_buy_volume"),
            pl.col("taker_buy_quote_volume").sum().alias("taker_buy_quote_volume"),
        ])
        .explode([
            "start_dt", "open", "high", "low", "close", "volume",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"
        ])
        .with_columns([
            pl.col("start_dt").dt.epoch("ms").alias("start_time"),
            (pl.col("start_dt").dt.epoch("ms") + interval_ms - 1).alias("end_time"),
        ])
        .select([
            "start_time", "end_time", "symbol", 
            "open", "high", "low", "close", "volume",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"
        ])
        .sort(["start_time", "symbol"])
    )
    
    return resampled
```

**Step 4: 執行測試確認通過**

執行：`pytest tests/data/test_loader_klines.py::TestLoadKlines -v`

預期：所有測試通過

**Step 5: 提交實作**

```bash
git add src/factorium/data/loader.py
git commit -m "feat: add klines direct loading support

- Detect data_type='klines' and bypass aggregation
- Load klines OHLCV data directly from Parquet
- Support resampling to different intervals (5m, 1h, 1d, etc.)
- Add _load_klines_direct() and _resample_klines() methods"
```

---

### Task 3: 添加 resample 功能測試

**Files:**
- Modify: `tests/data/test_loader_klines.py`

**Step 1: 寫一個失敗的測試 - klines resample**

在 `tests/data/test_loader_klines.py` 的 `TestLoadKlines` 類別中添加：

```python
def test_klines_resample_to_5m(self, sample_klines_data):
    """Test resampling 1m klines to 5m."""
    loader = BinanceDataLoader(base_path=sample_klines_data)

    with patch.object(loader, "_check_all_symbols_exist", return_value=True):
        result_1m = loader.load_aggbar(
            symbols=["BTCUSDT"],
            data_type="klines",
            market_type="futures",
            futures_type="um",
            start_date="2024-01-01",
            days=1,
            interval=60_000,  # 1m
            use_cache=False,
        )

        result_5m = loader.load_aggbar(
            symbols=["BTCUSDT"],
            data_type="klines",
            market_type="futures",
            futures_type="um",
            start_date="2024-01-01",
            days=1,
            interval=300_000,  # 5m
            use_cache=False,
        )

    # 5m should have 1/5 the bars of 1m
    assert len(result_5m) == len(result_1m) // 5


def test_klines_resample_to_1h(self, sample_klines_data):
    """Test resampling 1m klines to 1h."""
    loader = BinanceDataLoader(base_path=sample_klines_data)

    with patch.object(loader, "_check_all_symbols_exist", return_value=True):
        result_1m = loader.load_aggbar(
            symbols=["BTCUSDT"],
            data_type="klines",
            market_type="futures",
            futures_type="um",
            start_date="2024-01-01",
            days=1,
            interval=60_000,  # 1m
            use_cache=False,
        )

        result_1h = loader.load_aggbar(
            symbols=["BTCUSDT"],
            data_type="klines",
            market_type="futures",
            futures_type="um",
            start_date="2024-01-01",
            days=1,
            interval=3_600_000,  # 1h
            use_cache=False,
        )

    # 1h should have 1/60 the bars of 1m
    assert len(result_1h) == len(result_1m) // 60


def test_klines_raises_on_non_time_bar_type(self, sample_klines_data):
    """Test that klines raises error for non-time bar types."""
    loader = BinanceDataLoader(base_path=sample_klines_data)

    with patch.object(loader, "_check_all_symbols_exist", return_value=True):
        with pytest.raises(ValueError, match="only supports bar_type='time'"):
            loader.load_aggbar(
                symbols=["BTCUSDT"],
                data_type="klines",
                market_type="futures",
                futures_type="um",
                start_date="2024-01-01",
                days=1,
                bar_type="tick",  # Should fail
                interval=1000,
                use_cache=False,
            )
```

**Step 2: 執行測試確認通過**

執行：`pytest tests/data/test_loader_klines.py::TestLoadKlines -v`

預期：所有測試通過

**Step 3: 提交測試**

```bash
git add tests/data/test_loader_klines.py
git commit -m "test: add klines resample tests"
```

---

### Task 4: 更新文檔

**Files:**
- Modify: `docs/user-guide/data-acquisition.md` (if exists) or relevant doc

**Step 1: 添加 klines 使用範例**

在相關文檔中添加：

```markdown
### Loading Klines Data

Klines data is already aggregated OHLCV data from Binance. Unlike trades/aggTrades, 
klines don't require bar aggregation and are loaded directly.

```python
from factorium.data import BinanceDataLoader

loader = BinanceDataLoader()

# Load 1m klines (default)
agg = loader.load_aggbar(
    symbols=["BTCUSDT", "ETHUSDT"],
    data_type="klines",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=7,
)

# Resample to 5m
agg_5m = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="klines",
    market_type="futures",
    start_date="2024-01-01",
    days=7,
    interval=300_000,  # 5 minutes
)

# Resample to 1h
agg_1h = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="klines",
    market_type="futures",
    start_date="2024-01-01",
    days=7,
    interval=3_600_000,  # 1 hour
)
\```

**Notes:**
- Klines only supports `bar_type="time"` (the default)
- Downloaded data is always 1m, resampling happens on-the-fly
- Klines bypasses `BarAggregator` for better performance
```

**Step 2: 提交文檔**

```bash
git add docs/user-guide/data-acquisition.md
git commit -m "docs: add klines loading examples"
```

---

### Task 5: 執行完整測試並驗證

**Files:**
- N/A (verification step)

**Step 1: 執行所有相關測試**

執行：`pytest tests/data/ -v`

預期：所有測試通過

**Step 2: 手動驗證 klines 載入**

創建簡單腳本測試：

```python
from factorium.data import BinanceDataLoader

loader = BinanceDataLoader()

# Test klines loading
agg = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="klines",
    market_type="futures",
    start_date="2024-01-01",
    days=1,
)

print(f"Loaded {len(agg)} bars")
print(f"Columns: {agg.cols}")
print(agg.info())
```

預期：成功載入並顯示正確資訊

**Step 3: 確認 issue 已解決**

執行：`git log --oneline`

預期：看到所有相關 commits

---

## 完成檢查清單

- [ ] Task 1: 添加 klines 測試 (test_loader_klines.py)
- [ ] Task 2: 實作 klines 直接載入邏輯 (_load_klines_direct, _resample_klines)
- [ ] Task 3: 添加 resample 功能測試
- [ ] Task 4: 更新文檔
- [ ] Task 5: 執行完整測試並驗證
- [ ] 所有測試通過
- [ ] Issue #14 已解決
