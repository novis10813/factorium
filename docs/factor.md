## Factor 模組 (`factorium/factors/core.py`)

## 概述

`Factor` 是 Factorium 的核心資料結構之一，用來表示「多標的時間序列因子」。  
它通常由 `AggBar` 提供的 OHLCV 資料轉換而來，每一列代表：

- **start_time**: 該 bar 的開始時間（毫秒 timestamp）
- **end_time**: 該 bar 的結束時間（毫秒 timestamp）
- **symbol**: 標的代碼（例如 `BTCUSDT`）
- **factor**: 該因子的數值（例如 close 價、報酬率、Z-score 等）

`Factor` 在內部基於 `pandas.DataFrame`，並提供：

- **時間序列運算子**：`ts_mean`、`ts_std`、`ts_zscore`、`ts_delta`…  
- **橫截面運算子**：`rank`、`mean`、`median`  
- **數學運算子**：`abs`、`log`、`pow`、`signed_log1p`…  
- **繪圖功能**：透過 `.plot()` 快速視覺化因子

---

## 建立 Factor

### 從 AggBar 提取欄位

```python
from factorium import AggBar

agg = AggBar.from_df(df)      # 或使用 BinanceDataLoader.load_aggbar 建立
close = agg["close"]          # type: Factor
volume = agg["volume"]        # type: Factor
```

### 從 DataFrame 建立

```python
from factorium import Factor
import pandas as pd

df = pd.DataFrame({
    "start_time": start_ts,   # 毫秒 timestamp
    "end_time": end_ts,
    "symbol": symbols,
    "my_factor": values,
})

factor = Factor(df, name="my_factor")
```

### 從檔案載入

```python
from factorium import Factor

factor = Factor("factors/momentum.csv", name="momentum")
factor = Factor("factors/momentum.parquet", name="momentum")
```

---

## 資料結構與屬性

- **`factor.name`**: 因子名稱（字串）
- **`factor.data`**: 內部使用的 `pd.DataFrame`，欄位固定為  
  `["start_time", "end_time", "symbol", "factor"]`

```python
print(factor.name)
print(factor.data.head())
```

---

## 基本運算

### 算術與比較運算

```python
close = agg["close"]
volume = agg["volume"]

returns = (close - close.ts_shift(1)) / close.ts_shift(1)
vwap = (close * volume) / volume

is_up = close > close.ts_shift(1)            # 回傳 0/1 的 Factor
is_high_volume = volume > volume.ts_mean(20)
```

`Factor` 支援一般的運算子重載：

- **算術**：`+`、`-`、`*`、`/`
- **比較**：`<`、`<=`、`>`、`>=`、`==`、`!=`

---

## 時間序列運算（每個 symbol 各自計算）

常用方法（完整列表見 `README.md` 中的「時間序列運算子」）：

- **`ts_mean(window)`**: 滾動平均
- **`ts_std(window)`**: 滾動標準差
- **`ts_delta(period)`**: 差分
- **`ts_shift(period)`**: 時間位移（lag）
- **`ts_zscore(window)`**: Z-score 標準化
- **`ts_rank(window)`**: 視窗內排名

### 範例：動量與波動度

```python
close = agg["close"]

momentum_20 = close.ts_delta(20) / close.ts_shift(20)
volatility_20 = close.ts_std(20) / close.ts_mean(20)

risk_adjusted_momentum = momentum_20 / volatility_20
```

---

## 橫截面運算（同一時間點跨標的）

- **`rank()`**: 每個時間點對所有標的做百分位排名
- **`mean()`**: 每個時間點的橫截面平均
- **`median()`**: 每個時間點的橫截面中位數

### 範例：市場調整後報酬

```python
close = agg["close"]
returns = close.ts_delta(1) / close.ts_shift(1)

market_return = returns.mean()
excess_return = returns - market_return

momentum = close.ts_delta(20) / close.ts_shift(20)
momentum_rank = momentum.rank()
```

---

## 數學運算

常用方法：

- **`abs()`**、**`sign()`**、**`inverse()`**
- **`log(base=None)`**、`ln()`
- **`sqrt()`**、`pow(exp)`、`signed_pow(exp)`
- **`signed_log1p()`**：保留符號的 `log(1 + |x|)`
- **`max(other)`**、`min(other)`、`where(cond, other)`、`reverse()`

### 範例：處理極端值與符號

```python
returns = close.ts_delta(1) / close.ts_shift(1)

# 限制報酬在 [-10%, 10%]
capped = returns.max(-0.1).min(0.1)

# 保留符號的平方根
signed_sqrt = returns.signed_pow(0.5)
```

---

## 因子繪圖 (`Factor.plot`)

`Factor` 內建 `.plot()` 方法，使用 `matplotlib` 繪圖，底層由 `FactorPlotter` 類別實作。  
支援多種圖表類型與篩選參數。

### 介面說明

```python
fig = factor.plot(
    plot_type: str = "timeseries",          # 'timeseries' | 'heatmap' | 'distribution'
    symbols: list[str] | None = None,       # 要繪製的 symbols（None = 全部）
    start_time: datetime | None = None,     # 起始時間（datetime）
    end_time: datetime | None = None,       # 結束時間（datetime）
    figsize: tuple[int, int] = (12, 6),     # 圖片尺寸
    **kwargs,                                # 傳遞給各種繪圖函數的額外參數
)
```

### 圖表類型

- **`timeseries`**：時間序列線圖，每個 symbol 一條線
- **`heatmap`**：時間 × symbol 的因子值熱力圖
- **`distribution`**：分布圖  
  - 透過 `dist_type='histogram'` 或 `dist_type='density'` 控制型態

### 範例：時間序列圖

```python
from factorium import AggBar

agg = loader.load_aggbar(...)
close = agg["close"]

# 所有標的時間序列
fig = close.plot(plot_type="timeseries", figsize=(14, 6))

# 只看特定標的
fig = close.plot(
    plot_type="timeseries",
    symbols=["BTCUSDT", "ETHUSDT"],
    figsize=(12, 6),
)
```

### 範例：熱力圖

```python
fig = close.plot(
    plot_type="heatmap",
    figsize=(16, 8),
)
```

### 範例：分布圖

```python
# Histogram
fig = close.plot(
    plot_type="distribution",
    dist_type="histogram",
)

# Density / KDE
fig = close.plot(
    plot_type="distribution",
    dist_type="density",
)
```

### 範例：時間範圍與 symbol 篩選

```python
from datetime import datetime

fig = close.plot(
    plot_type="timeseries",
    symbols=["BTCUSDT"],
    start_time=datetime(2025, 1, 1, 9),
    end_time=datetime(2025, 1, 3, 18),
)
```

---

## 因子評估 (`Factor.eval`)

`Factor` 內建簡單的因子評估流程，底層由 `FactorEvaluator` 類別實作。  
透過 `factor.eval(...)`，可以一次計算常見的評估指標並輸出視覺化報告。

### 介面說明

```python
results = factor.eval(
    prices: Factor,                 # 價格因子（例如 close 價）
    periods: list[int] = (1, 5, 10),# 持有期（單位：bar 數 / 天等）
    quantiles: int = 5,             # 分層數量（global quantile）
    save_path: str | None = None,   # 若提供路徑，會輸出評估圖表 PNG
    **kwargs,
)
```

`results` 回傳一個 `dict`，主要包含：

- **`ic_series`**: `DataFrame`  
  - index: `end_time`  
  - columns: `IC_1d`, `IC_5d`, …（各持有期的每日 Rank IC）
- **`ic_mean`**: `Series` — 各持有期的平均 IC
- **`ic_ir`**: `Series` — 各持有期的 IC information ratio (`mean / std`)
- **`turnover_series`**: `Series` — 每日因子 turnover（橫截面 rank 自相關）
- **`turnover_mean`**: `float` — turnover 的平均值
- **`layer_returns`**: `dict[int, Series]`  
  - key: 持有期（例如 `1`, `5`, `10`）  
  - value: 該持有期下，各 **global quantile** 的平均未來報酬
- **`spread`**: `dict[int, float]` — 每個持有期下，最高 quantile 減最低 quantile 的 long-short spread

若指定 `save_path`，會額外輸出一張 **4 宮格圖表**：

- **左上：Rank IC over Time**
  - x 軸：時間
  - y 軸：IC  
  - 每一條線對應一個持有期（`IC_1d`, `IC_5d`, ...），可以觀察 IC 在時間上的 regime / 穩定度。
- **右上：Mean Layer Returns**
  - 取「最長持有期」的 `layer_returns`，畫出各 quantile 的平均未來報酬 bar 圖。  
  - quantile 是針對「全樣本的 factor 值」做 global `qcut` 後計算。
- **左下：IC Distribution (KDE by Period)**
  - 對 `ic_series` 中每個持有期的 IC 時間序列做 KDE，疊在同一張圖上。  
  - 可比較不同持有期的 IC 分布形狀與偏態。
- **右下：Factor Turnover (Rank Autocorrelation)**
  - 每日橫截面因子排名與前一日排名的相關係數。  
  - 接近 1 代表排名很穩定（低 turnover），接近 0 或負值代表排名變動較大（高 turnover）。

### 範例：對自訂因子進行評估

```python
from factorium import AggBar, Factor

agg = loader.load_aggbar(...)
close = agg["close"]          # 價格因子

# 建立一個簡單的動量因子
returns_1d = close.ts_delta(1) / close.ts_shift(1)
momentum_20 = close.ts_delta(20) / close.ts_shift(20)
factor = momentum_20

# 跑評估並輸出圖表
results = factor.eval(
    prices=close,
    periods=[1, 5, 20],
    quantiles=10,
    save_path="factor_eval_report.png",
)

print(results["ic_mean"])
print(results["spread"])
```

---

## 小結

- `Factor` 提供統一的因子表示方式與運算介面  
- 可從 `AggBar`、`DataFrame` 或檔案建立  
- 支援豐富的時間序列、橫截面與數學運算  
- 透過 `.plot()` 能快速視覺化多標的因子行為  

搭配 `BinanceDataLoader` + `Bar` + `AggBar`，可以很方便地從原始交易資料一路建構到完整的因子研究流程。


