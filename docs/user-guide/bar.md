# Bar 聚合模組

## 概述

Factorium 使用 DuckDB 進行高效能的 Bar 聚合，支援四種類型的 K 線：

| Bar 類型 | 聚合依據 | 適用場景 |
|----------|----------|----------|
| **Time Bar** | 固定時間間隔 | 傳統技術分析，時間一致性 |
| **Tick Bar** | 固定交易筆數 | 高頻交易分析，反映市場活動強度 |
| **Volume Bar** | 固定成交量 | 成交量分析，標準化成交量資訊 |
| **Dollar Bar** | 固定交易金額 | 跨價格比較，不受價格變化影響 |

---

## 使用方式

### 透過 `load_aggbar` 統一介面

最簡單的方式是透過 `BinanceDataLoader.load_aggbar()` 方法，它會自動處理下載、快取和聚合：

```python
from factorium import BinanceDataLoader

loader = BinanceDataLoader()

# Time Bar：1 分鐘 K 線
agg = loader.load_aggbar(
    symbols=["BTCUSDT", "ETHUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=7,
    bar_type="time",
    interval=60_000,  # 毫秒
)

# Tick Bar：每 1000 筆交易
tick_agg = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=1,
    bar_type="tick",
    interval=1000,
)

# Volume Bar：每累積 100 單位成交量
volume_agg = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=1,
    bar_type="volume",
    interval=100,
)

# Dollar Bar：每累積 100 萬美元
dollar_agg = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=1,
    bar_type="dollar",
    interval=1_000_000,
)
```

### `load_aggbar` 參數說明

| 參數 | 類型 | 說明 |
|------|------|------|
| `symbols` | `list[str]` | 要載入的標的代碼列表 |
| `data_type` | `str` | 資料類型（`"aggTrades"`, `"trades"` 等） |
| `market_type` | `str` | 市場類型（`"spot"`, `"futures"`） |
| `futures_type` | `str` | 期貨類型（`"um"` USDT-M, `"cm"` Coin-M） |
| `start_date` | `str` | 開始日期（`"YYYY-MM-DD"` 格式） |
| `days` | `int` | 載入天數 |
| `bar_type` | `str` | Bar 類型（`"time"`, `"tick"`, `"volume"`, `"dollar"`） |
| `interval` | `int` | 聚合間隔（意義依 bar_type 而定） |
| `use_cache` | `bool` | 是否使用快取（預設 `True`） |

---

## 輸出格式

聚合後的 `AggBar` 包含以下欄位：

| 欄位 | 類型 | 說明 |
|------|------|------|
| `symbol` | `str` | 標的代碼 |
| `start_time` | `int` | Bar 開始時間（毫秒 timestamp） |
| `end_time` | `int` | Bar 結束時間（毫秒 timestamp） |
| `open` | `float` | 開盤價 |
| `high` | `float` | 最高價 |
| `low` | `float` | 最低價 |
| `close` | `float` | 收盤價 |
| `volume` | `float` | 成交量 |
| `trade_count` | `int` | 交易筆數 |

### 存取資料

```python
# 查看 metadata（不需重新計算）
print(agg.symbols)           # ['BTCUSDT', 'ETHUSDT']
print(agg.metadata.num_rows) # 資料列數
print(agg.metadata.min_time) # 最早時間
print(agg.metadata.max_time) # 最晚時間

# 取得 Polars DataFrame
df_polars = agg.to_polars()

# 取得 Pandas DataFrame（用於繪圖等）
df_pandas = agg.to_df()

# 提取單一欄位為 Factor
close = agg["close"]
volume = agg["volume"]
```

---

## Bar 類型詳解

### Time Bar

按固定時間間隔聚合，是最常見的 K 線類型。

**特點：**
- 時間軸均勻分布
- 適合傳統技術分析
- 在低波動時期可能產生許多無交易的空 bar

**interval 單位：** 毫秒

```python
# 1 分鐘 K 線
agg_1m = loader.load_aggbar(..., bar_type="time", interval=60_000)

# 5 分鐘 K 線
agg_5m = loader.load_aggbar(..., bar_type="time", interval=300_000)

# 1 小時 K 線
agg_1h = loader.load_aggbar(..., bar_type="time", interval=3_600_000)
```

### Tick Bar

按固定交易筆數聚合。

**特點：**
- 反映市場活動強度
- 在高波動時期 bar 較密集
- 適合高頻交易研究

**interval 單位：** 交易筆數

```python
# 每 500 筆交易
agg = loader.load_aggbar(..., bar_type="tick", interval=500)

# 每 1000 筆交易
agg = loader.load_aggbar(..., bar_type="tick", interval=1000)
```

### Volume Bar

按固定成交量聚合。

**特點：**
- 標準化每個 bar 的成交量資訊
- 在大額交易時 bar 較密集
- 適合成交量分析策略

**interval 單位：** 成交量（與原始資料的 quantity 欄位單位相同）

```python
# 每累積 100 BTC
agg = loader.load_aggbar(..., bar_type="volume", interval=100)

# 每累積 1000 BTC
agg = loader.load_aggbar(..., bar_type="volume", interval=1000)
```

### Dollar Bar

按固定交易金額聚合（價格 × 成交量）。

**特點：**
- 不受價格變化影響
- 適合跨時期比較
- 在大額交易時 bar 較密集

**interval 單位：** 美元金額

```python
# 每累積 10 萬美元
agg = loader.load_aggbar(..., bar_type="dollar", interval=100_000)

# 每累積 100 萬美元
agg = loader.load_aggbar(..., bar_type="dollar", interval=1_000_000)
```

---

## 快取機制

`load_aggbar` 預設啟用快取，聚合後的資料會存為 Parquet 檔案：

```
~/.factorium/cache/bars/{market_type}/{futures_type}/{bar_type}/{symbol}/{date}.parquet
```

### 快取控制

```python
# 強制重新聚合（忽略快取）
agg = loader.load_aggbar(..., use_cache=False)
```

---

## 底層 API：BarAggregator

如果需要更細緻的控制，可以直接使用 `BarAggregator`。這是 `load_aggbar` 內部使用的聚合引擎：

```python
from factorium.data import BarAggregator
from factorium.data.adapters.base import ColumnMapping

aggregator = BarAggregator()

# 定義欄位映射
column_mapping = ColumnMapping(
    timestamp_col="transact_time",
    price_col="price",
    volume_col="quantity",
    symbol_col="symbol",
    is_buyer_maker_col="is_buyer_maker",
)

# 聚合 Time Bar
df, metadata = aggregator.aggregate_time_bars(
    parquet_pattern="/path/to/data/*.parquet",
    symbols=["BTCUSDT"],
    interval_ms=60_000,
    start_ts=1704067200000,  # 毫秒時間戳
    end_ts=1704153600000,
    column_mapping=column_mapping,
    include_buyer_seller=True,
)

# 聚合 Tick Bar
df, metadata = aggregator.aggregate_tick_bars(
    parquet_pattern="/path/to/data/*.parquet",
    symbols=["BTCUSDT"],
    interval_ticks=1000,
    start_ts=1704067200000,
    end_ts=1704153600000,
    column_mapping=column_mapping,
    include_buyer_seller=True,
)

# 聚合 Volume Bar
df, metadata = aggregator.aggregate_volume_bars(
    parquet_pattern="/path/to/data/*.parquet",
    symbols=["BTCUSDT"],
    interval_volume=100,
    start_ts=1704067200000,
    end_ts=1704153600000,
    column_mapping=column_mapping,
    include_buyer_seller=True,
)

# 聚合 Dollar Bar
df, metadata = aggregator.aggregate_dollar_bars(
    parquet_pattern="/path/to/data/*.parquet",
    symbols=["BTCUSDT"],
    interval_dollar=1_000_000,
    start_ts=1704067200000,
    end_ts=1704153600000,
    column_mapping=column_mapping,
    include_buyer_seller=True,
)
```

### 回傳值

所有 `aggregate_*_bars` 方法回傳 `tuple[pl.DataFrame, AggBarMetadata]`：

- **`pl.DataFrame`**: Polars DataFrame，包含聚合後的 OHLCV 資料
- **`AggBarMetadata`**: 包含 `symbols`, `min_time`, `max_time`, `num_rows`

> **注意**：對於大多數使用場景，建議直接使用 `BinanceDataLoader.load_aggbar()`，它會自動處理欄位映射、時間範圍計算等細節。

---

## 效能說明

Bar 聚合使用 DuckDB 的 SQL 引擎，效能特點：

- **記憶體效率**：透過 Parquet 格式的 lazy evaluation
- **並行處理**：DuckDB 自動使用多核心
- **零複製**：DuckDB → Polars 直接傳輸，無中間轉換

---

## 注意事項

1. **時間戳格式**：所有時間都是毫秒級 Unix timestamp
2. **最後一個 Bar**：Volume/Dollar Bar 的最後一個 bar 可能不完整（未達閾值）
3. **空資料處理**：若指定日期無資料，會回傳空的 DataFrame 和零值 metadata
4. **多標的聚合**：每個標的獨立聚合，最後合併
