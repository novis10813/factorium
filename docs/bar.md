# Bar 聚合模組 (`factorium/bar.py`)

## 概述

`bar.py` 提供了一組用於將逐筆交易數據（tick data）聚合成不同類型 K 線（bars）的類別。支援時間條、Tick 條、成交量條和美元條等多種聚合方式，並提供靈活的特徵工程功能。

## 類別繼承結構

```
BaseBar (ABC)
├── TimeBar      # 時間條
├── TickBar      # Tick 條
├── VolumeBar    # 成交量條
└── DollarBar    # 美元條
```

## 基礎類別：`BaseBar`

### 初始化參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `df` | `pd.DataFrame` | - | 原始交易數據 |
| `timestamp_col` | `str` | `'ts_init'` | 時間戳欄位名稱 |
| `price_col` | `str` | `'price'` | 價格欄位名稱 |
| `volume_col` | `str` | `'size'` | 成交量欄位名稱 |
| `interval` | `int` | `100000` | 聚合間隔（具體含義依子類別而定） |

### 屬性

| 屬性 | 類型 | 說明 |
|------|------|------|
| `bars` | `pd.DataFrame` | 聚合後的 K 線數據 |

### 輸出欄位

聚合後的 `bars` DataFrame 包含以下欄位：

| 欄位 | 說明 |
|------|------|
| `start_time` | Bar 開始時間 |
| `end_time` | Bar 結束時間 |
| `open` | 開盤價 |
| `high` | 最高價 |
| `low` | 最低價 |
| `close` | 收盤價 |
| `volume` | 成交量 |

### 主要方法

#### `apply()`

對已聚合的 bars 進行特徵計算。

```python
def apply(self, transformations: Dict[str, Callable]) -> 'BaseBar'
```

**參數說明：**

| 參數 | 類型 | 說明 |
|------|------|------|
| `transformations` | `Dict` | 特徵轉換字典，key 為新欄位名，value 為轉換函數 |

轉換函數接收完整的 `bars` DataFrame，回傳 `pd.Series` 或純量值。

**範例：**

```python
bar.apply({
    'forward_return_5': lambda bars: (bars['close'].shift(-5) - bars['close']) / bars['close'],
    'sma_20': lambda bars: bars['close'].rolling(20).mean(),
    'volatility': lambda bars: bars['close'].pct_change().rolling(20).std(),
    'price_momentum': lambda bars: bars['close'] / bars['close'].shift(10) - 1
})
```

---

## 子類別

### `TimeBar`

按固定時間間隔聚合交易數據。

#### 初始化參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `df` | `pd.DataFrame` | - | 原始交易數據 |
| `timestamp_col` | `str` | `'ts_init'` | 時間戳欄位名稱 |
| `price_col` | `str` | `'price'` | 價格欄位名稱 |
| `volume_col` | `str` | `'size'` | 成交量欄位名稱 |
| `interval_ms` | `int` | `60_000` | 時間間隔（毫秒） |

#### 使用範例

```python
from factorium import TimeBar

# 建立 1 分鐘 K 線
time_bar = TimeBar(
    df=trades_df,
    timestamp_col='time',
    price_col='price',
    volume_col='qty',
    interval_ms=60_000  # 1 分鐘
)

# 建立 5 分鐘 K 線
time_bar_5m = TimeBar(df=trades_df, interval_ms=300_000)

# 取得聚合後的 bars
bars = time_bar.bars
```

---

### `TickBar`

按固定筆數聚合交易數據。

#### 初始化參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `df` | `pd.DataFrame` | - | 原始交易數據 |
| `timestamp_col` | `str` | `'ts_init'` | 時間戳欄位名稱 |
| `price_col` | `str` | `'price'` | 價格欄位名稱 |
| `volume_col` | `str` | `'size'` | 成交量欄位名稱 |
| `interval_ticks` | `int` | `100000` | 每個 bar 的交易筆數 |

#### 使用範例

```python
from factorium import TickBar

# 每 1000 筆交易聚合成一個 bar
tick_bar = TickBar(
    df=trades_df,
    timestamp_col='time',
    price_col='price',
    volume_col='qty',
    interval_ticks=1000
)

bars = tick_bar.bars
```

---

### `VolumeBar`

按固定成交量聚合交易數據。

#### 初始化參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `df` | `pd.DataFrame` | - | 原始交易數據 |
| `timestamp_col` | `str` | `'time'` | 時間戳欄位名稱 |
| `price_col` | `str` | `'price'` | 價格欄位名稱 |
| `volume_col` | `str` | `'quote_qty'` | 成交量欄位名稱 |
| `interval_volume` | `int` | `100000` | 每個 bar 的目標成交量 |

#### 使用範例

```python
from factorium import VolumeBar

# 每累積 10000 單位成交量聚合成一個 bar
volume_bar = VolumeBar(
    df=trades_df,
    timestamp_col='time',
    price_col='price',
    volume_col='qty',
    interval_volume=10000
)

bars = volume_bar.bars
```

---

### `DollarBar`

按固定美元金額聚合交易數據（價格 × 成交量）。

#### 初始化參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `df` | `pd.DataFrame` | - | 原始交易數據 |
| `timestamp_col` | `str` | `'ts_init'` | 時間戳欄位名稱 |
| `price_col` | `str` | `'price'` | 價格欄位名稱 |
| `volume_col` | `str` | `'size'` | 成交量欄位名稱 |
| `interval_dollar` | `int` | `100000` | 每個 bar 的目標金額 |

#### 使用範例

```python
from factorium import DollarBar

# 每累積 100 萬美元聚合成一個 bar
dollar_bar = DollarBar(
    df=trades_df,
    timestamp_col='time',
    price_col='price',
    volume_col='qty',
    interval_dollar=1_000_000
)

bars = dollar_bar.bars
```

---

## 完整使用範例

### 基本使用

```python
import pandas as pd
from factorium import TimeBar, TickBar, VolumeBar, DollarBar

# 假設 trades_df 為原始交易數據
trades_df = pd.read_csv('trades.csv')

# 建立 1 分鐘 K 線
time_bar = TimeBar(
    df=trades_df,
    timestamp_col='time',
    price_col='price',
    volume_col='qty',
    interval_ms=60_000
)

print(time_bar.bars.head())
```

### 方法鏈式呼叫（Method Chaining）

```python
from factorium import TimeBar

# 建立 bar 並添加多種特徵
bars = (
    TimeBar(df=trades_df, interval_ms=60_000)
    .apply({
        'sma_20': lambda bars: bars['close'].rolling(20).mean(),
        'ema_12': lambda bars: bars['close'].ewm(span=12).mean(),
        'volatility': lambda bars: bars['close'].pct_change().rolling(20).std(),
        'forward_return': lambda bars: bars['close'].shift(-1) / bars['close'] - 1
    })
    .bars
)
```

### 不同 Bar 類型比較

```python
from factorium import TimeBar, TickBar, VolumeBar, DollarBar

# 同一份數據建立不同類型的 bar
time_bars = TimeBar(df=trades_df, interval_ms=60_000).bars
tick_bars = TickBar(df=trades_df, interval_ticks=1000).bars
volume_bars = VolumeBar(df=trades_df, interval_volume=10000).bars
dollar_bars = DollarBar(df=trades_df, interval_dollar=1_000_000).bars

print(f"TimeBar count: {len(time_bars)}")
print(f"TickBar count: {len(tick_bars)}")
print(f"VolumeBar count: {len(volume_bars)}")
print(f"DollarBar count: {len(dollar_bars)}")
```

---

## Bar 類型比較

| 類型 | 聚合依據 | 適用場景 | 優點 |
|------|----------|----------|------|
| `TimeBar` | 固定時間間隔 | 傳統技術分析 | 時間一致性，易於理解 |
| `TickBar` | 固定交易筆數 | 高頻交易分析 | 反映市場活動強度 |
| `VolumeBar` | 固定成交量 | 成交量分析 | 標準化成交量資訊 |
| `DollarBar` | 固定交易金額 | 跨價格比較 | 不受價格變化影響 |

---

## 效能優化

本模組使用 `numba` JIT 編譯來加速分組計算：

- `TimeBar._group_trades_by_time()`
- `TickBar._group_trades_by_tick()`
- `VolumeBar._group_trades_by_volume()`
- `DollarBar._group_trades_by_dollar()`

首次執行時會有編譯延遲，後續執行會顯著加速。

---

## 依賴套件

- `pandas`：數據處理
- `numpy`：數值運算
- `numba`：JIT 編譯加速

---

## 注意事項

- 輸入的 `df` 會被複製，不會修改原始數據
- `apply()` 回傳 `self`，支援方法鏈式呼叫
- 時間戳應為毫秒級別（milliseconds）
- `VolumeBar` 和 `DollarBar` 當累積值達到閾值時會切換到下一個 bar，最後一個 bar 可能不完整
- 各子類別的預設欄位名稱不同，使用時請注意對應
