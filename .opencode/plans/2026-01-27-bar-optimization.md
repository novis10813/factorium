# Bar Optimization Implementation Plan (Revised v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 擴充 `load_aggbar` 支援 TickBar/VolumeBar/DollarBar，統一 API，並移除 `bar.py`。

**Architecture:** 
- 使用 DuckDB SQL 聚合所有 bar 類型
- TimeBar 支援多標的，其他 bar 類型僅支援單標的
- 原 `bar.py` 移至 `tests/_legacy_bar/` 作為驗證基準

**Tech Stack:** Python, DuckDB, Polars, pytest

**預估時間:** 6-8 小時（包含完整測試驗證，考慮 legacy 對齊與邊界案例）

---

## 📐 Bar 定義規格（詳細版）

### 通用規則

| 項目 | 規格 |
|------|------|
| **單筆交易拆分** | **不拆分**。單筆交易完整歸屬於一個 bar（即使跨越多個門檻） |
| **邊界歸屬** | 累積到門檻那筆**歸屬當前 bar**，下一筆開始新 bar |
| **排序 Tie-breaker** | 同 timestamp 時使用 `(ts, price, volume, is_buyer_maker)` 保證順序穩定 |
| **VWAP 除以零** | 當 `SUM(volume) <= 1e-10` 時回傳 `NULL` |
| **interval 型別** | `float`（統一，因 volume/dollar 可能是小數） |
| **空 bar 處理** | 不產生空 bar（只有實際有交易的 bar 才輸出） |

### 各 Bar 類型定義

| Bar Type | 切分條件 | bar_id 計算 | 備註 |
|----------|----------|-------------|------|
| **TimeBar** | 固定時間間隔 | `(ts - start_time) // interval_ms` | 可能產生間隙 |
| **TickBar** | 固定 tick 數 | `(row_num - 1) // interval_ticks` | 連續無間隙 |
| **VolumeBar** | 累積成交量 | 見下方數學推導 | 連續無間隙 |
| **DollarBar** | 累積美元量 | 見下方數學推導 | 連續無間隙 |

### 🔬 Legacy vs DuckDB SQL 數學對應證明

**Legacy VolumeBar 邏輯（Numba）：**

```python
# 狀態機邏輯
current_volume = 0
bar_id = 0
for i in range(n):
    current_volume += volume[i]    # 1. 累加
    bar_ids[i] = bar_id            # 2. 指定（歸屬當前 bar）
    if current_volume >= threshold:
        current_volume = 0         # 3. 清零
        bar_id += 1                # 4. 換 bar
```

**DuckDB SQL 等價公式：**

```sql
-- cum_volume = 累積到這筆（含）的 volume
bar_id = FLOOR((cum_volume - volume) / threshold)
```

**數學證明：**

設 `v[i]` 為第 i 筆交易的 volume，`C[i] = Σv[0..i]` 為累積 volume。

Legacy 的 bar_id 實際上是「在處理第 i 筆**之前**已經完成了幾輪累積」：
- 第 i 筆歸屬的 bar_id = 在 `v[i]` 加入前，已經有幾次 `cum >= threshold`
- 這等價於 `FLOOR((C[i-1]) / threshold)` = `FLOOR((C[i] - v[i]) / threshold)`

**邊界案例驗證：**

| 案例 | volume[] | threshold | Legacy bar_ids | SQL bar_ids | 一致? |
|------|----------|-----------|----------------|-------------|-------|
| 基本 | [10,10,10,10] | 25 | [0,0,0,1] | [0,0,0,1] | ✅ |
| 剛好 | [10,10,10] | 30 | [0,0,0] | [0,0,0] | ✅ |
| 跨越 | [5,35,5] | 10 | [0,0,1] | [0,0,1] | ✅ |
| 大單 | [5,100,5] | 30 | [0,0,1] | [0,0,1] | ✅ |

> ⚠️ **關鍵洞察**：公式 `FLOOR((cum - volume) / threshold)` 等價於 legacy 的狀態機邏輯。
> 這是因為 legacy 的「清零」行為本質上是計算「到目前為止跨過了幾個門檻」。

---

## 🔧 DuckDB SQL 設計原則

### 1. 順序保證策略（關鍵修正）

**問題：** `FIRST()/LAST()` 在 `GROUP BY` 後**不保證順序**。DuckDB 的計算計畫可能因檔案分段、並行處理等因素導致結果不穩定。

**解決方案：** 使用 `ARG_MIN/ARG_MAX` 搭配序列號（seq）：

```sql
-- Step 1: 在 numbered CTE 中產生唯一序列號（使用完整 tie-breaker）
ROW_NUMBER() OVER (ORDER BY ts, price, volume, is_buyer_maker) AS seq

-- Step 2: 聚合時使用 ARG_MIN/ARG_MAX
ARG_MIN(ts, seq) AS start_time,
ARG_MAX(ts, seq) AS end_time,
ARG_MIN(price, seq) AS open,
ARG_MAX(price, seq) AS close,
```

### 2. Tie-breaker 策略（同 timestamp 處理）

**問題：** 同一個 timestamp 可能有多筆交易，需要確保排序穩定。

**解決方案：** 使用複合排序（與 Legacy 一致）：

```sql
-- 優先順序：ts → price → volume → is_buyer_maker
ORDER BY ts, price, volume, is_buyer_maker
```

> ⚠️ 若原始資料有 `trade_id` 或 `agg_trade_id`，應優先使用該欄位作為 tie-breaker。

### 3. VWAP 安全除法

```sql
CASE 
    WHEN SUM(volume) <= 1e-10 THEN NULL 
    ELSE SUM(price * volume) / SUM(volume) 
END AS vwap
```

> 使用 `1e-10` 作為 EPSILON，與專案 `constants.py` 一致。

### 4. SQL 參數化（避免注入）

使用 escape 函數避免 SQL 注入：

```python
def escape_sql_string(s: str) -> str:
    """Escape single quotes in SQL string to prevent injection."""
    return s.replace("'", "''")

# 使用方式
escaped_symbol = escape_sql_string(symbol)
query = f"... WHERE symbol = '{escaped_symbol}' ..."
```

> 未來可考慮改用 DuckDB 參數化查詢，但目前 escape 足夠應付內部工具需求。

---

## Phase 0: 修復現有 aggregate_time_bars 的順序問題

### Task 0: 修正 aggregate_time_bars 使用 ARG_MIN/ARG_MAX

**Files:**
- Modify: `src/factorium/data/aggregator.py`

**Changes:**
將現有的 `FIRST(price) AS open, LAST(price) AS close` 改為：

```sql
-- 加入序列號
ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY {ts_col}) AS seq

-- 聚合使用 ARG_MIN/ARG_MAX
ARG_MIN(price, seq) AS open,
ARG_MAX(price, seq) AS close,
ARG_MIN(ts, seq) AS first_ts,
ARG_MAX(ts, seq) AS last_ts,
```

**Step: Commit**

```bash
git add src/factorium/data/aggregator.py
git commit -m "fix(aggregator): use ARG_MIN/ARG_MAX to guarantee OHLC order"
```

---

## Phase 1: 保留 Legacy Bar 作為測試基準

### Task 1: 建立 tests/_legacy_bar 目錄

**Files:**
- Create: `tests/_legacy_bar/__init__.py`
- Create: `tests/_legacy_bar/bar.py` (複製自 `src/factorium/bar.py`)

**Step 1: 建立目錄結構**

```bash
mkdir -p tests/_legacy_bar
```

**Step 2: 複製 bar.py 到 _legacy_bar**

```bash
cp src/factorium/bar.py tests/_legacy_bar/bar.py
```

**Step 3: 建立 __init__.py**

```python
"""Legacy bar implementations for testing DuckDB aggregation correctness."""

from .bar import BaseBar, TimeBar, TickBar, VolumeBar, DollarBar

__all__ = ["BaseBar", "TimeBar", "TickBar", "VolumeBar", "DollarBar"]
```

**Step 4: Commit**

```bash
git add tests/_legacy_bar/
git commit -m "test: move bar.py to _legacy_bar for aggregation verification"
```

---

## Phase 2: 擴充 BarAggregator

### Task 2: 實作 aggregate_tick_bars

**Files:**
- Modify: `src/factorium/data/aggregator.py`

**DuckDB SQL 邏輯：**
1. 使用 `ROW_NUMBER() OVER (ORDER BY ts)` 為每筆 tick 編號（作為 seq）
2. `FLOOR((seq - 1) / interval_ticks)` 計算 bar_id
3. `GROUP BY bar_id` 聚合，使用 `ARG_MIN/ARG_MAX` 保證順序

```python
def aggregate_tick_bars(
    self,
    parquet_pattern: str,
    symbol: str,
    interval_ticks: int,
    column_mapping: ColumnMapping,
    include_buyer_seller: bool = True,
) -> pd.DataFrame:
    """Aggregate tick data into tick-based OHLCV bars.

    Args:
        parquet_pattern: Glob pattern for Parquet files
        symbol: Single symbol (tick bars don't align across symbols)
        interval_ticks: Number of ticks per bar
        column_mapping: Column name mapping for the data source
        include_buyer_seller: Include buyer/seller statistics

    Returns:
        DataFrame with OHLCV columns
    """
    ts_col = column_mapping.timestamp
    price_col = column_mapping.price
    volume_col = column_mapping.volume
    ibm_col = column_mapping.is_buyer_maker
    
    # Escape symbol to prevent SQL injection
    escaped_symbol = symbol.replace("'", "''")

    buyer_seller_sql = ""
    buyer_seller_cols = ""
    if include_buyer_seller and ibm_col:
        buyer_seller_sql = """
            , SUM(CASE WHEN NOT is_buyer_maker THEN 1 ELSE 0 END) AS num_buyer
            , SUM(CASE WHEN is_buyer_maker THEN 1 ELSE 0 END) AS num_seller
            , SUM(CASE WHEN NOT is_buyer_maker THEN volume ELSE 0 END) AS num_buyer_volume
            , SUM(CASE WHEN is_buyer_maker THEN volume ELSE 0 END) AS num_seller_volume
        """
        buyer_seller_cols = ", num_buyer, num_seller, num_buyer_volume, num_seller_volume"

    query = f"""
        WITH raw_data AS (
            SELECT
                symbol,
                {ts_col} AS ts,
                {price_col} AS price,
                {volume_col} AS volume
                {f", {ibm_col} AS is_buyer_maker" if include_buyer_seller and ibm_col else ""}
            FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
            WHERE symbol = '{escaped_symbol}'
        ),
        numbered AS (
            SELECT
                *,
                ROW_NUMBER() OVER (ORDER BY ts) AS seq,
                (ROW_NUMBER() OVER (ORDER BY ts) - 1) // {interval_ticks} AS bar_id
            FROM raw_data
        ),
        aggregated AS (
            SELECT
                symbol,
                bar_id,
                ARG_MIN(ts, seq) AS start_time,
                ARG_MAX(ts, seq) AS end_time,
                ARG_MIN(price, seq) AS open,
                MAX(price) AS high,
                MIN(price) AS low,
                ARG_MAX(price, seq) AS close,
                SUM(volume) AS volume,
                CASE 
                    WHEN SUM(volume) <= 1e-10 THEN NULL 
                    ELSE SUM(price * volume) / SUM(volume) 
                END AS vwap
                {buyer_seller_sql}
            FROM numbered
            GROUP BY symbol, bar_id
        )
        SELECT
            symbol, start_time, end_time,
            open, high, low, close, volume, vwap
            {buyer_seller_cols}
        FROM aggregated
        ORDER BY bar_id
    """

    try:
        return duckdb.query(query).df()
    except duckdb.IOException as e:
        logger.warning(f"DuckDB tick bar aggregation failed: {e}")
        return pd.DataFrame()
```

---

### Task 3: 實作 aggregate_volume_bars

**Files:**
- Modify: `src/factorium/data/aggregator.py`

**DuckDB SQL 邏輯：**

Volume bar 的關鍵是要匹配 legacy 的邏輯：
- 累積成交量
- 當累積 >= 門檻時，**當前這筆仍屬於舊 bar**，下一筆開始新 bar

```python
def aggregate_volume_bars(
    self,
    parquet_pattern: str,
    symbol: str,
    interval_volume: float,
    column_mapping: ColumnMapping,
    include_buyer_seller: bool = True,
) -> pd.DataFrame:
    """Aggregate tick data into volume-based OHLCV bars.
    
    Bar boundary rule: When cumulative volume >= threshold, the current trade
    belongs to the current bar, and the next trade starts a new bar.
    """
    ts_col = column_mapping.timestamp
    price_col = column_mapping.price
    volume_col = column_mapping.volume
    ibm_col = column_mapping.is_buyer_maker
    
    escaped_symbol = symbol.replace("'", "''")

    buyer_seller_sql = ""
    buyer_seller_cols = ""
    if include_buyer_seller and ibm_col:
        buyer_seller_sql = """
            , SUM(CASE WHEN NOT is_buyer_maker THEN 1 ELSE 0 END) AS num_buyer
            , SUM(CASE WHEN is_buyer_maker THEN 1 ELSE 0 END) AS num_seller
            , SUM(CASE WHEN NOT is_buyer_maker THEN volume ELSE 0 END) AS num_buyer_volume
            , SUM(CASE WHEN is_buyer_maker THEN volume ELSE 0 END) AS num_seller_volume
        """
        buyer_seller_cols = ", num_buyer, num_seller, num_buyer_volume, num_seller_volume"

    # Note: This SQL replicates the legacy behavior:
    # - current_volume += volume; bar_ids[i] = bar_id; if current_volume >= threshold: bar_id += 1
    # The key insight is that we need to count how many times the threshold was crossed
    # BEFORE this trade (not including this trade's volume contribution to crossing).
    #
    # We use a cumulative sum approach where bar_id is determined by the cumulative
    # volume at the START of each trade (cum_volume - volume).
    
    query = f"""
        WITH raw_data AS (
            SELECT
                symbol,
                {ts_col} AS ts,
                {price_col} AS price,
                {volume_col} AS volume
                {f", {ibm_col} AS is_buyer_maker" if include_buyer_seller and ibm_col else ""}
            FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
            WHERE symbol = '{escaped_symbol}'
        ),
        numbered AS (
            SELECT
                *,
                ROW_NUMBER() OVER (ORDER BY ts) AS seq
            FROM raw_data
        ),
        cumulative AS (
            SELECT
                *,
                SUM(volume) OVER (ORDER BY seq) AS cum_volume
            FROM numbered
        ),
        with_bar_id AS (
            SELECT
                *,
                -- bar_id based on cumulative volume BEFORE adding this trade
                -- This matches legacy: assign bar_id first, then check threshold
                FLOOR((cum_volume - volume) / {interval_volume})::BIGINT AS bar_id
            FROM cumulative
        ),
        aggregated AS (
            SELECT
                symbol,
                bar_id,
                ARG_MIN(ts, seq) AS start_time,
                ARG_MAX(ts, seq) AS end_time,
                ARG_MIN(price, seq) AS open,
                MAX(price) AS high,
                MIN(price) AS low,
                ARG_MAX(price, seq) AS close,
                SUM(volume) AS volume,
                CASE 
                    WHEN SUM(volume) <= 1e-10 THEN NULL 
                    ELSE SUM(price * volume) / SUM(volume) 
                END AS vwap
                {buyer_seller_sql}
            FROM with_bar_id
            GROUP BY symbol, bar_id
        )
        SELECT
            symbol, start_time, end_time,
            open, high, low, close, volume, vwap
            {buyer_seller_cols}
        FROM aggregated
        ORDER BY bar_id
    """

    try:
        return duckdb.query(query).df()
    except duckdb.IOException as e:
        logger.warning(f"DuckDB volume bar aggregation failed: {e}")
        return pd.DataFrame()
```

---

### Task 4: 實作 aggregate_dollar_bars

**Files:**
- Modify: `src/factorium/data/aggregator.py`

**DuckDB SQL 邏輯：**
類似 Volume Bar，但累積 `price * volume`（dollar volume）

```python
def aggregate_dollar_bars(
    self,
    parquet_pattern: str,
    symbol: str,
    interval_dollar: float,
    column_mapping: ColumnMapping,
    include_buyer_seller: bool = True,
) -> pd.DataFrame:
    """Aggregate tick data into dollar-volume based OHLCV bars.
    
    Bar boundary rule: Same as volume bars, but threshold is dollar volume.
    """
    ts_col = column_mapping.timestamp
    price_col = column_mapping.price
    volume_col = column_mapping.volume
    ibm_col = column_mapping.is_buyer_maker
    
    escaped_symbol = symbol.replace("'", "''")

    buyer_seller_sql = ""
    buyer_seller_cols = ""
    if include_buyer_seller and ibm_col:
        buyer_seller_sql = """
            , SUM(CASE WHEN NOT is_buyer_maker THEN 1 ELSE 0 END) AS num_buyer
            , SUM(CASE WHEN is_buyer_maker THEN 1 ELSE 0 END) AS num_seller
            , SUM(CASE WHEN NOT is_buyer_maker THEN volume ELSE 0 END) AS num_buyer_volume
            , SUM(CASE WHEN is_buyer_maker THEN volume ELSE 0 END) AS num_seller_volume
        """
        buyer_seller_cols = ", num_buyer, num_seller, num_buyer_volume, num_seller_volume"

    query = f"""
        WITH raw_data AS (
            SELECT
                symbol,
                {ts_col} AS ts,
                {price_col} AS price,
                {volume_col} AS volume,
                {price_col} * {volume_col} AS dollar_volume
                {f", {ibm_col} AS is_buyer_maker" if include_buyer_seller and ibm_col else ""}
            FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
            WHERE symbol = '{escaped_symbol}'
        ),
        numbered AS (
            SELECT
                *,
                ROW_NUMBER() OVER (ORDER BY ts) AS seq
            FROM raw_data
        ),
        cumulative AS (
            SELECT
                *,
                SUM(dollar_volume) OVER (ORDER BY seq) AS cum_dollar
            FROM numbered
        ),
        with_bar_id AS (
            SELECT
                *,
                FLOOR((cum_dollar - dollar_volume) / {interval_dollar})::BIGINT AS bar_id
            FROM cumulative
        ),
        aggregated AS (
            SELECT
                symbol,
                bar_id,
                ARG_MIN(ts, seq) AS start_time,
                ARG_MAX(ts, seq) AS end_time,
                ARG_MIN(price, seq) AS open,
                MAX(price) AS high,
                MIN(price) AS low,
                ARG_MAX(price, seq) AS close,
                SUM(volume) AS volume,
                CASE 
                    WHEN SUM(volume) <= 1e-10 THEN NULL 
                    ELSE SUM(price * volume) / SUM(volume) 
                END AS vwap
                {buyer_seller_sql}
            FROM with_bar_id
            GROUP BY symbol, bar_id
        )
        SELECT
            symbol, start_time, end_time,
            open, high, low, close, volume, vwap
            {buyer_seller_cols}
        FROM aggregated
        ORDER BY bar_id
    """

    try:
        return duckdb.query(query).df()
    except duckdb.IOException as e:
        logger.warning(f"DuckDB dollar bar aggregation failed: {e}")
        return pd.DataFrame()
```

**Step: Commit all aggregator changes**

```bash
git add src/factorium/data/aggregator.py
git commit -m "feat(aggregator): add tick/volume/dollar bar aggregation with correct ordering"
```

---

## Phase 3: 統一 load_aggbar API

### Task 5: 修改 BinanceDataLoader.load_aggbar

**Files:**
- Modify: `src/factorium/data/loader.py`

**Changes:**

1. **重命名** `load_aggbar_fast` → `load_aggbar`（移除舊的 `load_aggbar`）
2. **新增** `bar_type` 參數：`"time"` | `"tick"` | `"volume"` | `"dollar"`
3. **interval 型別改為 float**（統一處理 volume/dollar 的小數情況）
4. **嚴格檢查**：非 TimeBar 時，若 symbols 多於一個則拋出 ValueError
5. **路由到對應的 aggregator 方法**

```python
def load_aggbar(
    self,
    symbols: Union[str, List[str]],
    data_type: str,
    market_type: str,
    futures_type: str = "um",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
    interval: float = 60_000.0,
    bar_type: str = "time",
    force_download: bool = False,
    use_cache: bool = True,
) -> "AggBar":
    """Load bar data and return as AggBar.

    Args:
        symbols: Symbol(s) to load. For non-time bars, must be a single symbol.
        data_type: Type of data (e.g., "aggTrades", "trades")
        market_type: "spot" or "futures"
        futures_type: "um" or "cm" (for futures only)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        days: Number of days from today (alternative to start/end dates)
        interval: Bar interval (meaning depends on bar_type)
            - time: milliseconds (default 60_000 = 1 minute)
            - tick: number of ticks
            - volume: volume threshold (float)
            - dollar: dollar volume threshold (float)
        bar_type: "time", "tick", "volume", or "dollar"
        force_download: Force re-download even if files exist
        use_cache: Use cached aggregated bars if available (time bars only)

    Returns:
        AggBar with aggregated bar data

    Raises:
        ValueError: If bar_type is not "time" and multiple symbols provided
        ValueError: If bar_type is invalid
    """
    # Normalize symbols to list
    if isinstance(symbols, str):
        symbols = [symbols]

    # Validate bar_type
    valid_bar_types = {"time", "tick", "volume", "dollar"}
    if bar_type not in valid_bar_types:
        raise ValueError(f"bar_type must be one of {valid_bar_types}, got '{bar_type}'")
    
    # Validate: non-time bars only support single symbol
    if bar_type != "time" and len(symbols) > 1:
        raise ValueError(
            f"bar_type='{bar_type}' only supports single symbol, "
            f"got {len(symbols)} symbols: {symbols}"
        )

    # ... existing download logic ...

    # Route to appropriate aggregator method
    if bar_type == "time":
        df = self._aggregator.aggregate_time_bars(
            parquet_pattern=parquet_pattern,
            symbols=symbols,
            interval_ms=int(interval),
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )
    elif bar_type == "tick":
        df = self._aggregator.aggregate_tick_bars(
            parquet_pattern=parquet_pattern,
            symbol=symbols[0],
            interval_ticks=int(interval),
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )
    elif bar_type == "volume":
        df = self._aggregator.aggregate_volume_bars(
            parquet_pattern=parquet_pattern,
            symbol=symbols[0],
            interval_volume=float(interval),
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )
    elif bar_type == "dollar":
        df = self._aggregator.aggregate_dollar_bars(
            parquet_pattern=parquet_pattern,
            symbol=symbols[0],
            interval_dollar=float(interval),
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )

    return AggBar.from_df(df)
```

**Step: Commit**

```bash
git add src/factorium/data/loader.py
git commit -m "feat(loader): unify load_aggbar API with bar_type support"
```

---

## Phase 4: 移除舊的 bar.py（Breaking Change）

> ⚠️ **這是破壞性變更**：移除 `bar.py` 會影響任何 `from factorium import TimeBar` 的使用者。

### Task 6: 從 src/factorium 移除 bar.py

**Files:**
- Delete: `src/factorium/bar.py`
- Modify: `src/factorium/__init__.py` (移除 bar 相關 export)
- Modify: `src/factorium/aggbar.py` (移除 BaseBar 相關 import，如果有的話)
- Move: `tests/test_bar.py` → `tests/_legacy_bar/test_bar.py`

**相容性策略：**

由於這是內部工具，採用**直接移除**策略（不做 deprecation）：

1. **全 repo 搜尋**確認無其他引用：
   ```bash
   rg "from factorium.bar import|from factorium import.*Bar" --type py
   rg "factorium\.bar\." --type py
   ```

2. **更新 CHANGELOG**（如果有的話）記錄 breaking change

3. **提供遷移指南**：
   ```python
   # 舊用法（將被移除）
   from factorium import TimeBar, VolumeBar
   bar = VolumeBar(df, interval_volume=1000)
   
   # 新用法
   from factorium.data import load_aggbar
   aggbar = load_aggbar(symbols="BTCUSDT", bar_type="volume", interval=1000, ...)
   ```

**Step 1: 全 repo 搜尋引用**

```bash
rg "from factorium.bar import|from factorium import.*Bar|factorium\.bar\." --type py
```

處理所有找到的引用。

**Step 2: 更新 src/factorium/__init__.py**

移除以下 export：
```python
# 移除這些行
from .bar import BaseBar, TimeBar, TickBar, VolumeBar, DollarBar
```

更新 `__all__` 移除 Bar 類別。

**Step 3: 檢查 src/factorium/aggbar.py**

確認沒有 `from .bar import` 相關程式碼。

**Step 4: 移動 tests/test_bar.py**

```bash
mv tests/test_bar.py tests/_legacy_bar/test_bar.py
```

更新測試中的 import：
```python
# 舊
from factorium import TimeBar, VolumeBar
# 新
from tests._legacy_bar import TimeBar, VolumeBar
```

**Step 5: 刪除 bar.py**

```bash
rm src/factorium/bar.py
```

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor!: remove bar.py from factorium package (BREAKING CHANGE)

Migration: Use load_aggbar(bar_type='volume'|'tick'|'dollar') instead of
VolumeBar/TickBar/DollarBar classes.

Legacy bar classes are preserved in tests/_legacy_bar/ for verification."
```

---

## Phase 5: 測試與驗證

### Task 7: 建立 Aggregator 測試

**Files:**
- Create: `tests/data/test_aggregator_bars.py`

**測試策略（詳細版）：**

1. **完整欄位比對**：
   - OHLCV: open, high, low, close, volume
   - Timing: start_time, end_time
   - Derived: vwap
   - Buyer/Seller: num_buyer, num_seller, num_buyer_volume, num_seller_volume

2. **浮點精度策略（欄位分級）**：
   | 欄位類型 | 比對方式 | 參數 |
   |----------|----------|------|
   | Timestamp | exact | `assert_array_equal` |
   | Price (open/high/low/close) | relative | `rtol=1e-9` |
   | Volume | relative | `rtol=1e-9` |
   | VWAP | relative + NaN | `rtol=1e-9, equal_nan=True` |
   | Count (num_buyer/seller) | exact | `assert_array_equal` |

3. **邊界案例測試矩陣**：
   | 案例 | 描述 | 驗證重點 |
   |------|------|----------|
   | `test_exact_threshold` | 累積剛好等於門檻 | bar_id 正確換檔 |
   | `test_same_timestamp` | 同 ts 多筆 | 排序穩定、OHLC 正確 |
   | `test_large_single_trade` | 單筆跨多門檻 | 不拆分、歸屬正確 |
   | `test_zero_volume` | volume=0 的 bar | VWAP 為 NULL |
   | `test_single_trade` | 僅 1 筆交易 | edge case |
   | `test_many_small_trades` | 大量小單 | 效能 + 正確性 |

4. **多檔案一致性測試**：
   - 同資料分成 2/3/5 個 parquet，結果必須相同
   - 驗證 DuckDB 並行讀取不影響順序

5. **人造可推導案例**（非隨機）：
   ```python
   # 可手算驗證的固定案例
   def test_deterministic_volume_bars():
       trades = pd.DataFrame({
           "ts_init": [1, 2, 3, 4, 5],
           "price": [100, 101, 102, 103, 104],
           "size": [10, 10, 10, 10, 10],  # cum: 10,20,30,40,50
       })
       # interval=25: bar0=[0,1], bar1=[2,3], bar2=[4]
       # 手算：
       # bar0: open=100, close=101, high=101, low=100, volume=20
       # bar1: open=102, close=103, high=103, low=102, volume=20
       # bar2: open=104, close=104, volume=10
   ```

```python
"""Tests for BarAggregator comparing with legacy bar implementations."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from factorium.data.aggregator import BarAggregator
from factorium.data.adapters.base import ColumnMapping
from tests._legacy_bar import TickBar, VolumeBar, DollarBar


# === Fixtures ===

@pytest.fixture
def column_mapping() -> ColumnMapping:
    return ColumnMapping(
        timestamp="ts_init",
        price="price",
        volume="size",
        is_buyer_maker="is_buyer_maker",
    )


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """Create sample trade data with deterministic values."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "symbol": ["BTCUSDT"] * n,
        "ts_init": np.arange(n) * 100 + 1700000000000,
        "price": 50000 + np.random.randn(n).cumsum() * 10,
        "size": np.abs(np.random.randn(n)) * 0.1 + 0.01,
        "is_buyer_maker": np.random.choice([True, False], n),
    })


@pytest.fixture
def edge_case_trades() -> pd.DataFrame:
    """Trades with edge cases: same timestamp, exact threshold crossing."""
    return pd.DataFrame({
        "symbol": ["BTCUSDT"] * 10,
        "ts_init": [1000, 1000, 1000, 2000, 2000, 3000, 4000, 5000, 5000, 6000],
        "price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        "size": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "is_buyer_maker": [True, False, True, False, True, False, True, False, True, False],
    })


@pytest.fixture
def deterministic_trades() -> pd.DataFrame:
    """Trades with hand-calculable expected results."""
    return pd.DataFrame({
        "symbol": ["BTCUSDT"] * 5,
        "ts_init": [1000, 2000, 3000, 4000, 5000],
        "price": [100.0, 101.0, 102.0, 103.0, 104.0],
        "size": [10.0, 10.0, 10.0, 10.0, 10.0],  # cum: 10, 20, 30, 40, 50
        "is_buyer_maker": [True, False, True, False, True],
    })


# === Helper Functions ===

def assert_bars_equal(
    duckdb_df: pd.DataFrame, 
    legacy_df: pd.DataFrame, 
    check_cols: list = None,
    rtol: float = 1e-9,
):
    """Assert two bar DataFrames are equal within tolerance.
    
    Uses field-specific comparison strategies:
    - Timestamps: exact match
    - Counts: exact match  
    - Prices/Volumes: relative tolerance
    - VWAP: relative tolerance with NaN handling
    """
    if check_cols is None:
        check_cols = ["open", "high", "low", "close", "volume", "vwap", "start_time", "end_time"]
    
    assert len(duckdb_df) == len(legacy_df), f"Length mismatch: {len(duckdb_df)} vs {len(legacy_df)}"
    
    for col in check_cols:
        if col not in duckdb_df.columns or col not in legacy_df.columns:
            continue
        
        duckdb_vals = duckdb_df[col].values
        legacy_vals = legacy_df[col].values
        
        if col in ["start_time", "end_time", "num_buyer", "num_seller"]:
            # Exact match for timestamps and counts
            np.testing.assert_array_equal(duckdb_vals, legacy_vals, err_msg=f"{col} mismatch")
        elif col == "vwap":
            # Handle NaN/None in vwap
            np.testing.assert_allclose(
                duckdb_vals, legacy_vals, rtol=rtol, equal_nan=True, err_msg=f"{col} mismatch"
            )
        else:
            # Float comparison with tolerance
            np.testing.assert_allclose(
                duckdb_vals, legacy_vals, rtol=rtol, err_msg=f"{col} mismatch"
            )


# === Tick Bar Tests ===

class TestTickBarAggregation:
    def test_tick_bar_matches_legacy(self, sample_trades, column_mapping, tmp_path):
        """DuckDB tick bar aggregation should match legacy implementation."""
        parquet_path = tmp_path / "trades.parquet"
        sample_trades.to_parquet(parquet_path)

        aggregator = BarAggregator()
        duckdb_result = aggregator.aggregate_tick_bars(
            parquet_pattern=str(parquet_path),
            symbol="BTCUSDT",
            interval_ticks=100,
            column_mapping=column_mapping,
        )

        legacy_bar = TickBar(
            sample_trades,
            timestamp_col="ts_init",
            price_col="price",
            volume_col="size",
            interval_ticks=100,
        )
        legacy_result = legacy_bar.bars

        assert_bars_equal(duckdb_result, legacy_result)

    def test_tick_bar_same_timestamp(self, edge_case_trades, column_mapping, tmp_path):
        """Tick bars should handle same-timestamp trades correctly."""
        parquet_path = tmp_path / "trades.parquet"
        edge_case_trades.to_parquet(parquet_path)

        aggregator = BarAggregator()
        duckdb_result = aggregator.aggregate_tick_bars(
            parquet_pattern=str(parquet_path),
            symbol="BTCUSDT",
            interval_ticks=5,
            column_mapping=column_mapping,
        )

        legacy_bar = TickBar(
            edge_case_trades,
            timestamp_col="ts_init",
            price_col="price",
            volume_col="size",
            interval_ticks=5,
        )
        legacy_result = legacy_bar.bars

        assert_bars_equal(duckdb_result, legacy_result)


# === Volume Bar Tests ===

class TestVolumeBarAggregation:
    def test_volume_bar_matches_legacy(self, sample_trades, column_mapping, tmp_path):
        """DuckDB volume bar aggregation should match legacy implementation."""
        parquet_path = tmp_path / "trades.parquet"
        sample_trades.to_parquet(parquet_path)

        aggregator = BarAggregator()
        duckdb_result = aggregator.aggregate_volume_bars(
            parquet_pattern=str(parquet_path),
            symbol="BTCUSDT",
            interval_volume=5.0,
            column_mapping=column_mapping,
        )

        legacy_bar = VolumeBar(
            sample_trades,
            timestamp_col="ts_init",
            price_col="price",
            volume_col="size",
            interval_volume=5.0,
        )
        legacy_result = legacy_bar.bars

        assert_bars_equal(duckdb_result, legacy_result)

    def test_volume_bar_exact_threshold(self, edge_case_trades, column_mapping, tmp_path):
        """Volume bar should correctly handle exact threshold crossing."""
        parquet_path = tmp_path / "trades.parquet"
        edge_case_trades.to_parquet(parquet_path)

        aggregator = BarAggregator()
        duckdb_result = aggregator.aggregate_volume_bars(
            parquet_pattern=str(parquet_path),
            symbol="BTCUSDT",
            interval_volume=30.0,
            column_mapping=column_mapping,
        )

        legacy_bar = VolumeBar(
            edge_case_trades,
            timestamp_col="ts_init",
            price_col="price",
            volume_col="size",
            interval_volume=30.0,
        )
        legacy_result = legacy_bar.bars

        assert_bars_equal(duckdb_result, legacy_result)

    def test_volume_bar_deterministic(self, deterministic_trades, column_mapping, tmp_path):
        """Volume bar with hand-calculable expected results."""
        parquet_path = tmp_path / "trades.parquet"
        deterministic_trades.to_parquet(parquet_path)

        aggregator = BarAggregator()
        result = aggregator.aggregate_volume_bars(
            parquet_pattern=str(parquet_path),
            symbol="BTCUSDT",
            interval_volume=25.0,  # bar0: [0,1] (20), bar1: [2,3] (20), bar2: [4] (10)
            column_mapping=column_mapping,
        )
        
        # bar_id = FLOOR((cum - vol) / 25)
        # trade 0: (10-10)/25 = 0 → bar 0
        # trade 1: (20-10)/25 = 0 → bar 0
        # trade 2: (30-10)/25 = 0 → bar 0 (cum_before=20)
        # trade 3: (40-10)/25 = 1 → bar 1 (cum_before=30)
        # trade 4: (50-10)/25 = 1 → bar 1 (cum_before=40)
        
        assert len(result) == 2, f"Expected 2 bars, got {len(result)}"
        # Bar 0: trades 0,1,2 → open=100, close=102, high=102, low=100
        # Bar 1: trades 3,4 → open=103, close=104, high=104, low=103


# === Dollar Bar Tests ===

class TestDollarBarAggregation:
    def test_dollar_bar_matches_legacy(self, sample_trades, column_mapping, tmp_path):
        """DuckDB dollar bar aggregation should match legacy implementation."""
        parquet_path = tmp_path / "trades.parquet"
        sample_trades.to_parquet(parquet_path)

        aggregator = BarAggregator()
        duckdb_result = aggregator.aggregate_dollar_bars(
            parquet_pattern=str(parquet_path),
            symbol="BTCUSDT",
            interval_dollar=50000.0,
            column_mapping=column_mapping,
        )

        legacy_bar = DollarBar(
            sample_trades,
            timestamp_col="ts_init",
            price_col="price",
            volume_col="size",
            interval_dollar=50000.0,
        )
        legacy_result = legacy_bar.bars

        assert_bars_equal(duckdb_result, legacy_result)


# === Multi-file Consistency Tests ===

class TestMultiFileConsistency:
    def test_split_parquet_same_result(self, sample_trades, column_mapping, tmp_path):
        """Aggregation should be consistent whether data is in one or multiple files."""
        # Single file
        single_path = tmp_path / "single" / "trades.parquet"
        single_path.parent.mkdir()
        sample_trades.to_parquet(single_path)

        # Split into multiple files
        split_dir = tmp_path / "split"
        split_dir.mkdir()
        n = len(sample_trades)
        sample_trades.iloc[:n//2].to_parquet(split_dir / "part1.parquet")
        sample_trades.iloc[n//2:].to_parquet(split_dir / "part2.parquet")

        aggregator = BarAggregator()
        
        single_result = aggregator.aggregate_tick_bars(
            parquet_pattern=str(single_path),
            symbol="BTCUSDT",
            interval_ticks=100,
            column_mapping=column_mapping,
        )
        
        split_result = aggregator.aggregate_tick_bars(
            parquet_pattern=str(split_dir / "*.parquet"),
            symbol="BTCUSDT",
            interval_ticks=100,
            column_mapping=column_mapping,
        )

        assert_bars_equal(single_result, split_result)

    def test_three_way_split_consistency(self, sample_trades, column_mapping, tmp_path):
        """Verify consistency with 3-way file split."""
        split_dir = tmp_path / "split3"
        split_dir.mkdir()
        n = len(sample_trades)
        sample_trades.iloc[:n//3].to_parquet(split_dir / "part1.parquet")
        sample_trades.iloc[n//3:2*n//3].to_parquet(split_dir / "part2.parquet")
        sample_trades.iloc[2*n//3:].to_parquet(split_dir / "part3.parquet")

        # Compare against legacy (single DataFrame)
        legacy_bar = VolumeBar(
            sample_trades,
            timestamp_col="ts_init",
            price_col="price",
            volume_col="size",
            interval_volume=5.0,
        )
        
        aggregator = BarAggregator()
        split_result = aggregator.aggregate_volume_bars(
            parquet_pattern=str(split_dir / "*.parquet"),
            symbol="BTCUSDT",
            interval_volume=5.0,
            column_mapping=column_mapping,
        )

        assert_bars_equal(split_result, legacy_bar.bars)


# === Edge Case Tests ===

class TestEdgeCases:
    def test_single_trade(self, column_mapping, tmp_path):
        """Handle single trade edge case."""
        single_trade = pd.DataFrame({
            "symbol": ["BTCUSDT"],
            "ts_init": [1000],
            "price": [100.0],
            "size": [1.0],
            "is_buyer_maker": [True],
        })
        parquet_path = tmp_path / "single.parquet"
        single_trade.to_parquet(parquet_path)

        aggregator = BarAggregator()
        result = aggregator.aggregate_tick_bars(
            parquet_pattern=str(parquet_path),
            symbol="BTCUSDT",
            interval_ticks=10,
            column_mapping=column_mapping,
        )

        assert len(result) == 1
        assert result.iloc[0]["open"] == 100.0
        assert result.iloc[0]["close"] == 100.0
        assert result.iloc[0]["volume"] == 1.0

    def test_large_single_trade_crosses_multiple_thresholds(self, column_mapping, tmp_path):
        """Large single trade should belong to one bar (no splitting)."""
        trades = pd.DataFrame({
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "ts_init": [1000, 2000, 3000],
            "price": [100.0, 101.0, 102.0],
            "size": [5.0, 100.0, 5.0],  # Middle trade crosses multiple thresholds
            "is_buyer_maker": [True, False, True],
        })
        parquet_path = tmp_path / "large_trade.parquet"
        trades.to_parquet(parquet_path)

        aggregator = BarAggregator()
        duckdb_result = aggregator.aggregate_volume_bars(
            parquet_pattern=str(parquet_path),
            symbol="BTCUSDT",
            interval_volume=10.0,
            column_mapping=column_mapping,
        )

        legacy_bar = VolumeBar(
            trades,
            timestamp_col="ts_init",
            price_col="price",
            volume_col="size",
            interval_volume=10.0,
        )

        assert_bars_equal(duckdb_result, legacy_bar.bars)
```

**Step: Commit**

```bash
git add tests/data/test_aggregator_bars.py
git commit -m "test: add comprehensive bar aggregator tests with legacy comparison"
```

---

### Task 8: 執行完整測試套件

**Step 1: 執行所有測試**

```bash
pytest -v --tb=short
```

Expected: All tests PASS

**Step 2: 檢查是否有遺漏的 bar.py 引用**

```bash
rg "from factorium.bar import|from factorium import.*Bar|factorium\.bar\." --type py
```

Expected: 無輸出（除了 tests/_legacy_bar）

**Step 3: 檢查 AggBar 單標的相容性**

```python
# 驗證單標的 AggBar 可以正常使用 Factor API
from factorium import AggBar, Factor
import pandas as pd

df = pd.DataFrame({
    "symbol": ["BTCUSDT"] * 10,
    "start_time": range(10),
    "end_time": range(1, 11),
    "close": [100 + i for i in range(10)],
})
aggbar = AggBar.from_df(df)
factor = aggbar["close"]
print(factor.ts_mean(3).to_pandas())
```

---

## Summary

| Phase | Tasks | 目標 |
|-------|-------|------|
| 0 | Task 0 | 修復現有 aggregate_time_bars 的 FIRST/LAST 順序問題 |
| 1 | Task 1 | 保留 legacy bar 作為測試基準 |
| 2 | Task 2-4 | 擴充 BarAggregator 支援 tick/volume/dollar bars |
| 3 | Task 5 | 統一 load_aggbar API |
| 4 | Task 6 | 移除舊的 bar.py（**Breaking Change**） |
| 5 | Task 7-8 | 測試與驗證 |

**預估時間:** 6-8 小時（包含完整測試與 legacy 對齊）

**風險與緩解：**

| 風險 | 緩解措施 |
|------|----------|
| DuckDB `ARG_MIN/ARG_MAX` 版本差異 | 要求 DuckDB >= 0.9 |
| Volume/Dollar bar 邊界不一致 | 完整的邊界案例測試 + 數學證明 |
| 同 timestamp 排序不穩定 | 使用完整 tie-breaker `(ts, price, volume, is_buyer_maker)` |
| bar.py 移除是 breaking change | 全 repo 搜尋 + commit message 標注 |

**回滾策略：**
- 每個 Task 有獨立 commit，可以個別 revert
- Task 6 使用 `refactor!:` commit message 標注 breaking change
- Legacy bar 保留在 `tests/_legacy_bar/`，可隨時恢復

---

## Appendix A: 測試矩陣

| 測試類別 | 覆蓋項目 | 測試數量 |
|----------|----------|----------|
| **基本功能** | tick/volume/dollar bar 與 legacy 結果一致 | 4 |
| **欄位完整性** | open, high, low, close, volume, vwap, start_time, end_time | - |
| **邊界案例** | 剛好到門檻、同 timestamp 多筆、跨門檻、單筆交易 | 4 |
| **VWAP 安全** | volume=0 時 vwap 為 NULL | 1 |
| **多檔案一致性** | 單檔 vs 2-way/3-way 分割 parquet | 2 |
| **順序穩定性** | 同 timestamp 不同執行結果一致 | 1 |
| **人造案例** | 可手算驗證的固定值 | 1 |

總計: ~13+ 個測試案例

---

## Appendix B: AggBar 型別契約

### 輸入契約（from_df 期待的 DataFrame）

| 欄位 | 型別 | 必須 | 說明 |
|------|------|------|------|
| `symbol` | str | ✅ | 標的代碼 |
| `start_time` | int | ✅ | Bar 起始時間戳（ms） |
| `end_time` | int | ✅ | Bar 結束時間戳（ms） |
| `open` | float | ✅ | 開盤價 |
| `high` | float | ✅ | 最高價 |
| `low` | float | ✅ | 最低價 |
| `close` | float | ✅ | 收盤價 |
| `volume` | float | ✅ | 成交量 |
| `vwap` | float/None | ❌ | 成交量加權平均價 |
| `num_buyer` | int | ❌ | 買方交易筆數 |
| `num_seller` | int | ❌ | 賣方交易筆數 |
| `num_buyer_volume` | float | ❌ | 買方成交量 |
| `num_seller_volume` | float | ❌ | 賣方成交量 |

### 輸出保證

- 單標的 AggBar 可正常使用 Factor API（如 `ts_mean`, `cs_rank` 等）
- `symbol` 欄位始終存在（即使只有一個標的）
- 時間欄位為整數型別（毫秒時間戳）
