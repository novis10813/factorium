# 策略回測（Backtest）

本頁說明 Factorium 的回測模組，以及如何使用向量化回測器與權重約束來評估因子策略。

---

## 1. 核心概念

- **價格資料 (`prices`)**：使用 `AggBar` 表示的多標的 OHLCV 資料，欄位至少包含  
  `["start_time", "end_time", "symbol", "open", "high", "low", "close", "volume"]`。
- **信號因子 (`signal`)**：任何 `Factor` 物件，需與 `prices` 在 `end_time, symbol` 上對齊，通常為已做過橫截面處理的排名 / Z-score。
- **避免前視偏差**：回測時會使用「前一根 bar 的信號」在當前 bar 交易。
- **向量化實作**：內部全部以 Polars 向量化計算完成，再轉成 pandas 計算績效指標。

主要類別：

- `factorium.backtest.Backtester`：使用者面向的回測器（別名，實際指向 `VectorizedBacktester`）。
- `factorium.backtest.vectorized.VectorizedBacktester`：向量化回測實作。
- `factorium.backtest.vectorized.BacktestResult`：回測結果容器。

---

## 2. 最簡範例：從因子到回測

```python
from factorium import AggBar
from factorium.backtest import Backtester
import polars as pl

# 1. 載入聚合後資料
agg = AggBar.from_df(pl.read_parquet("data/multi_symbol.parquet"))

# 2. 構建因子信號
close = agg["close"]
momentum = (close.ts_delta(20) / close.ts_shift(20)).cs_rank()

# 3. 建立回測器並執行
bt = Backtester(
    prices=agg,
    signal=momentum,
    initial_capital=10_000.0,
    neutralization="market",   # "market" (市場中性) 或 "none" (long-only)
    transaction_cost=0.0003,   # 可為 float 或 (buy_rate, sell_rate) tuple
    frequency="1h",            # 用於年化指標
)
result = bt.run()

print(result.metrics)
```

---

## 3. BacktestResult 與資料格式

`BacktestResult` 為向量化回測器的結果型別，所有表格皆為 Polars DataFrame：

- **`equity_curve: pl.DataFrame`**  
  - 欄位：`["end_time", "total_value"]`
- **`returns: pl.DataFrame`**  
  - 欄位：`["end_time", "return"]`
- **`metrics: dict[str, float]`**  
  - 主要指標：
    - `total_return`
    - `annual_return`
    - `annual_volatility`
    - `sharpe_ratio`
    - `sortino_ratio`
    - `calmar_ratio`
    - `max_drawdown`
    - `win_rate`
- **`trades: pl.DataFrame`**  
  - 欄位：`["end_time", "symbol", "qty", "price", "cost"]`
- **`portfolio_history: pl.DataFrame`**  
  - 欄位：`["end_time", "cash", "market_value", "total_value"]`

如需 pandas 版本，可呼叫：

```python
pandas_result = result.to_pandas()
print(pandas_result.equity_curve.tail())
print(pandas_result.metrics)
```

---

## 4. 市場中性與 Long-only 權重

### 4.1 `neutralization="market"`（市場中性）

使用 cross-section demean + L1 normalize 的方式：

\[
w_{i,t} = \frac{x_{i,t} - \bar{x}_t}{\sum_j |x_{j,t} - \bar{x}_t|}
\]

其中：

- \(\sum_i w_{i,t} = 0\)
- \(\sum_i |w_{i,t}| = 1\)

程式中對應到 `backtest.utils.neutralize_weights_polars`，測試會檢查每個 `end_time` 權重和接近 0。

### 4.2 `neutralization="none"`（Long-only）

- 僅使用正號信號：`prev_signal > 0`。
- 每個時間點將正信號正規化為權重和 1，其餘權重為 0。

適合做「只做多」的策略。

---

## 5. 權重約束（Constraints）

向量化回測器支援在權重計算後，透過一系列 **Polars-based** 約束物件調整權重：

- **基底類別**：`factorium.backtest.constraints.WeightConstraint`
- **具體實作**：
  - `MaxPositionConstraint(max_weight: float)`：限制單一標的最大絕對權重。
  - `LongOnlyConstraint()`：將負權重設為 0。
  - `MaxGrossExposureConstraint(max_exposure: float)`：限制每個時間點的總絕對權重。
  - `MarketNeutralConstraint()`：強制每個時間點權重和為 0。

約束會依序套用在含有欄位 `["end_time", "symbol", "weight"]` 的 Polars DataFrame 上：

```python
from factorium.backtest import Backtester, MaxPositionConstraint, LongOnlyConstraint

constraints = [
    MaxPositionConstraint(max_weight=0.1),  # 單一標的不超過 10%
    LongOnlyConstraint(),                  # 強制不做空
]

bt = Backtester(
    prices=agg,
    signal=momentum,
    constraints=constraints,
    neutralization="market",  # 先做市場中性，再套用約束
)
result = bt.run()
```

> **注意**：約束只負責「限制」權重，不會自動重新正規化使權重和維持 1；若需要額外規則，可以自訂 `WeightConstraint` 子類別。

---

## 6. 與 ResearchSession 的整合

實務上，你通常不需要直接建立 `Backtester`，而是透過 `ResearchSession` 來串接資料、因子與回測：

```python
from factorium import ResearchSession

session = ResearchSession.from_parquet("data/multi_symbol.parquet")
signal = session.factor("close").ts_delta(20).cs_rank()

result = session.backtest(
    signal,
    neutralization="market",
    transaction_cost=0.0003,
)

print(result.metrics)
```

若你需要對不同參數組合進行多個回測，可以在同一個 `ResearchSession` 中重複呼叫 `backtest()`。

---

## 7. 典型工作流程總結

1. **準備資料**：使用 `BinanceDataLoader.load_aggbar()` 或 `AggBar.from_df()` 建立 `AggBar`。
2. **建立因子**：使用 `AggBar["close"]` 等欄位與 TS/CS 運算子構建 `Factor`。
3. **（可選）分析因子**：使用 `FactorAnalyzer` 或 `ResearchSession.analyze()` 檢查 IC / 分層收益。
4. **執行回測**：
   - 直接使用 `Backtester(prices=agg, signal=signal)`；或
   - 透過 `ResearchSession.backtest(signal)`。
5. **查看結果**：讀取 `BacktestResult.metrics`、`equity_curve`、`trades` 等欄位，或將結果轉成 pandas 作進一步分析。

