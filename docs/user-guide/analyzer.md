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
    *   使用 `Factor` 運算邏輯 (`ts_shift(-p)`) 進行高效計算。

### 2.2 繪圖類別 `FactorAnalyzerPlotter`
位於 `src/factorium/factors/plotting_analyzer.py`。

*   負責所有圖表的繪製工作，與分析邏輯分離。
*   基於 `matplotlib` 實作。
*   支援 IC 時序圖、IC 直方圖、分層收益柱狀圖、累積收益曲線。

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

## 4. 實作細節與安全性

*   **數據對齊**: `prepare_data` 採用 Inner Join + DropNA 策略，確保分析僅基於完整的數據點，避免偏差。
*   **依賴管理**: 繪圖功能被設計為選用 (Optional)，核心邏輯不強依賴 `matplotlib`，僅在呼叫 `plot_*` 方法時才需要。
*   **錯誤處理**: 針對空數據、無效的分層數量、相關性計算樣本不足等情況加入了防禦性檢查。
