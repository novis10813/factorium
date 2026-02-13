## Factor 模組 (`factorium/factors/core.py`)

## 概述

`Factor` 是 Factorium 的核心資料結構之一，用來表示「多標的時間序列因子」。  
它通常由 `AggBar` 提供的 OHLCV 資料轉換而來，每一列代表：

- **start_time**: 該 bar 的開始時間（毫秒 timestamp）
- **end_time**: 該 bar 的結束時間（毫秒 timestamp）
- **symbol**: 標的代碼（例如 `BTCUSDT`）
- **factor**: 該因子的數值（例如 close 價、報酬率、Z-score 等）

`Factor` 在內部以 **Polars LazyFrame** 作為主要計算表示（用於延遲計算與效能），並提供：

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
- **`factor.data`**: 以 `polars.DataFrame` 形式回傳目前資料（會觸發 collect），欄位固定為  
  `["start_time", "end_time", "symbol", "factor"]`
- **`factor.lazy`**: 以 `polars.LazyFrame` 形式回傳（不會立刻執行）
- **`factor.to_pandas()`**: 如需 pandas，請用此方法轉換

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
momentum_rank = momentum.cs_rank()
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

`Factor` 內建簡單的因子評估流程，底層由 `FactorAnalyzer` 類別實作。  
透過 `factor.eval(...)`，可以一次計算常見的評估指標並輸出視覺化報告。

### 介面說明

```python
result = factor.eval(
    prices: Factor | AggBar,        # 價格數據（Factor 或 AggBar）
    periods: int = 1,                # 持有期（單位：bar 數 / 天等，目前僅支援單一 int）
    quantiles: int = 5,              # 分層數量（per-day cross-sectional quantiles）
    output_dir: str | None = None,  # 若提供路徑，會輸出評估結果到時間戳目錄
    price_col: str = "close",       # 價格欄位名稱（當 prices 為 AggBar 時使用）
    mask: str | None = None,         # 選填：限制分析樣本的遮罩欄位名稱
    **kwargs,
) -> FactorAnalysisResult
```

### 與 Universe / Checklist 遮罩整合

若你在 `AggBar` 上已建立 Universe/Checklist 遮罩欄位，可以直接傳入 `mask=`：

```python
result = momentum_20.eval(
    prices=agg,
    periods=1,
    quantiles=5,
    mask="checklist_mask",
)
```

這可確保分位分組與後續評估都只在可交易資產池上進行，避免把不在策略範圍內的標的納入統計。

`result` 回傳一個 `FactorAnalysisResult` dataclass，主要包含：

- **`factor_name`**: 因子名稱
- **`periods`**: 持有期（`int`，未來可能支援 `list[int]`）
- **`quantiles`**: 分層數量
- **`ic_series`**: `pd.DataFrame` — 每日 Rank IC 時間序列（index=start_time）
- **`ic_summary`**: `dict` — IC 統計摘要（`mean_ic`, `ic_std`, `ic_ir`, `t-stat`）
- **`turnover_series`**: `pd.Series` — 每日因子 turnover（1 - rank autocorrelation，index=start_time）
- **`turnover_mean`**: `float` — turnover 的平均值
- **`quantile_returns`**: `pd.DataFrame` — 各 quantile 的平均未來報酬（MultiIndex: start_time, quantile）
- **`cumulative_returns`**: `pd.DataFrame | None` — 各 quantile 的累計報酬（可選）

`FactorAnalysisResult` 提供以下便利方法：

- **`to_dict()`**: 轉換為字典格式（向後相容）
- **`save(output_dir)`**: 保存完整實驗結果到指定目錄（見下方說明）

若指定 `output_dir`，會在該目錄下創建時間戳子目錄（格式：`YYYYMMDD_HHMMSS_{factor_name}/`），並輸出：

- **CSV 檔案**：`ic_series.csv`、`ic_summary.csv`、`turnover.csv`、`quantile_returns.csv`、`cumulative_returns.csv`
- **配置檔案**：`config.json`（包含實驗設定與 metadata）
- **圖表**：`plots/` 子目錄下的 PNG 圖表（IC 時間序列、IC 分佈、分層收益、累計收益）

### 範例：對自訂因子進行評估

#### 使用 Factor 作為價格數據

```python
from factorium import AggBar, Factor

agg = loader.load_aggbar(...)
close = agg["close"]          # 價格因子

# 建立一個簡單的動量因子
momentum_20 = close.ts_delta(20) / close.ts_shift(20)

# 跑評估並輸出結果
result = momentum_20.eval(
    prices=close,              # 傳入 Factor
    periods=1,
    quantiles=10,
    output_dir="./experiments",
)

print(result.ic_summary['mean_ic'])
print(result.turnover_mean)
```

#### 使用 AggBar 作為價格數據

```python
# 直接傳入 AggBar，會自動使用 price_col 指定的欄位（預設 "close"）
result = momentum_20.eval(
    prices=agg,                # 傳入 AggBar
    periods=1,
    quantiles=5,
    price_col="close",         # 指定價格欄位
    output_dir="./experiments",
)

# 查看結果摘要
print(result)
# FactorAnalysisResult: momentum_20
#   Periods: 1, Quantiles: 5
#   Mean IC: 0.0234
#   IC Std: 0.1456
#   IC IR: 0.1607
#   Turnover: 0.8234
```

#### 手動保存結果

```python
# 不指定 output_dir，先進行分析
result = momentum_20.eval(prices=agg, periods=1)

# 稍後再保存（例如根據條件決定是否保存）
if result.ic_summary['ic_ir'] > 1.0:
    result.save("./experiments")
```

---

## 小結

- `Factor` 提供統一的因子表示方式與運算介面  
- 可從 `AggBar`、`DataFrame` 或檔案建立  
- 支援豐富的時間序列、橫截面與數學運算  
- 透過 `.plot()` 能快速視覺化多標的因子行為  

搭配 `BinanceDataLoader` + `Bar` + `AggBar`，可以很方便地從原始交易資料一路建構到完整的因子研究流程。

