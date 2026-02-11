# 因子分析

本頁說明 `FactorAnalyzer` 的使用方式與功能。

## 1. 概述

`FactorAnalyzer` 是一個整合性的因子評估工具，它接收一個因子訊號 (`Factor`) 與價格數據 (`AggBar`)，並提供以下三大類分析功能：

1.  **IC 分析 (Information Coefficient)**: 評估因子預測能力。
2.  **分層收益分析 (Quantile Returns)**: 評估因子的單調性與獲利能力。
3.  **視覺化 (Plotting)**: 提供各項指標的圖表輸出。

## 2. 類別架構

### 2.1 核心類別 `FactorAnalyzer`
位於 `src/factorium/factors/analyzer.py`。

*   **初始化**: `FactorAnalyzer(factor: Factor, prices: AggBar | Factor)`
    *   `factor`: 待分析的因子物件。
    *   `prices`: 價格數據，支援 `AggBar` 或直接傳入價格 `Factor`。若傳入 `AggBar`，預設取用 `close` 欄位。

*   **數據準備**: `prepare_data(periods=[1, 5, 10], price_col="close")`
    *   這是執行分析前的必要步驟。
    *   負責計算未來收益率 (Forward Returns)。
    *   執行嚴格的數據對齊 (Inner Join)，去除無效數據。
    *   使用 Polars 向量化計算（`LazyFrame` + `shift(-p).over("symbol")`），效能與記憶體表現更佳。

*   **一鍵分析**: `analyze(price_col="close", periods=1) -> FactorAnalysisResult`
    *   自動呼叫 `prepare_data()`、`calculate_ic()`、`calculate_ic_summary()`、`calculate_quantile_returns()` 等方法。
    *   回傳結構化結果物件 `FactorAnalysisResult`，方便後續串接報告或序列化。

### 2.2 繪圖類別 `FactorAnalyzerPlotter`
位於 `src/factorium/factors/plotting_analyzer.py`。

*   負責所有圖表的繪製工作，與分析邏輯分離。
*   基於 `matplotlib` 實作。
*   支援 IC 時序圖、IC 直方圖、分層收益柱狀圖、累積收益曲線。

### 2.3 結構化結果 `FactorAnalysisResult`

`FactorAnalyzer.analyze()` 會回傳 `FactorAnalysisResult` dataclass，方便在 Notebook 或報告中使用：

```python
from factorium.factors import FactorAnalysisResult

result: FactorAnalysisResult = analyzer.analyze(price_col="close", periods=1)
```

主要欄位：

- `factor_name: str`：因子名稱。
- `periods: int`：分析使用的持有期（forward return horizon）。
- `quantiles: int`：使用的分位數數量。
- `ic_series: pd.DataFrame`：IC 時間序列（index 為 `start_time`）。
- `ic_summary: dict`：IC 摘要統計，包含 `mean_ic`、`ic_std`、`ic_ir`、`t-stat`。
- `quantile_returns: pd.DataFrame`：分層收益（MultiIndex: `start_time`, `quantile`）。
- `cumulative_returns: Optional[pd.DataFrame]`：各分層累積收益（若計算成功）。

同時提供：

- `to_dict()`：轉換為 `dict`，方便序列化或與舊版 API 相容。

## 3. 功能詳解

### 3.1 IC 分析 (IC Analysis)
*   **方法**: `calculate_ic(method='rank'|'normal')`
*   **邏輯**: 計算每一期因子值與未來收益率的截面相關係數。
    *   Rank IC: Spearman 相關係數 (預設)。
    *   Normal IC: Pearson 相關係數。
*   **摘要**: `calculate_ic_summary()` 提供 Mean IC, IC Std, IC IR (Mean/Std), t-stat 等統計數據。

### 3.2 分層收益分析 (Quantile Analysis)
*   **方法**: `calculate_quantile_returns(quantiles=5)`
*   **邏輯**: 每一期將股票依因子值分為 N 組 (Quantiles)。
*   **指標**:
    *   平均收益 (Mean Return): 各組的平均未來收益。
    *   累積收益 (Cumulative Return): 各組收益的複利累積曲線。
    *   多空對沖 (Long-Short): Top Quantile - Bottom Quantile 的收益曲線。
*   **穩健性**: 使用 `pd.qcut(duplicates='drop')` 處理因子值重複過多的情況。

### 3.3 繪圖 (Plotting)
*   `plot_ic(period, plot_type='ts'|'hist')`: 繪製 IC 走勢或分佈。
*   `plot_quantile_returns(period)`: 繪製各分層的平均收益。
*   `plot_cumulative_returns(period, long_short=True)`: 繪製分層累積收益曲線。

## 4. 完整使用範例

```python
from factorium import BinanceDataLoader
from factorium.factors import FactorAnalyzer

# 1. 載入資料
loader = BinanceDataLoader()
agg = loader.load_aggbar(
    symbols=["BTCUSDT", "ETHUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=30,
    bar_type="time",
    interval=60_000,
)

# 2. 建立因子
close = agg["close"]
momentum = (close.ts_delta(20) / close.ts_shift(20)).cs_rank()

# 3. 建立分析器
analyzer = FactorAnalyzer(
    factor=momentum,
    prices=agg,  # 傳入 AggBar，會自動使用 close 欄位
)

# 4. 一鍵分析（會自動呼叫 prepare_data）
result = analyzer.analyze(price_col="close", periods=1)
print(result.ic_summary)

# 5. 若需更細緻控制，可手動操作：
analyzer.prepare_data(periods=[1, 5, 10], price_col="close")
ic = analyzer.calculate_ic(method="rank")
ic_summary = analyzer.calculate_ic_summary(method="rank")
quantile_returns = analyzer.calculate_quantile_returns(quantiles=5, period=1)
cumulative_returns = analyzer.calculate_cumulative_returns(quantiles=5, period=1, long_short=True)

# 6. 繪製圖表
analyzer.plot_ic(period=1, method="rank", plot_type="ts")
analyzer.plot_quantile_returns(quantiles=5, period=1)
analyzer.plot_cumulative_returns(quantiles=5, period=1, long_short=True)
```

## 5. 實作細節與安全性

*   **數據對齊**: `prepare_data` 採用 Inner Join + DropNA 策略，確保分析僅基於完整的數據點，避免偏差。
*   **Polars 為主**: 所有對齊與 forward return 計算都在 Polars 中完成，再轉為 pandas 給 IC / 分層結果。
*   **依賴管理**: 繪圖功能被設計為選用 (Optional)，核心邏輯不強依賴 `matplotlib`，僅在呼叫 `plot_*` 方法時才需要。
*   **錯誤處理**: 針對空數據、無效的分層數量、相關性計算樣本不足等情況加入了防禦性檢查。
*   **必要步驟**: 若直接使用 `calculate_ic()`、`calculate_quantile_returns()` 等方法，必須先呼叫 `prepare_data()`；若使用 `analyze()`，會自動處理。
