# 策略回測（Backtest）

本頁說明 Factorium 的回測模組，以及如何透過 `AlphaPipeline` 將因子信號轉成權重、使用向量化回測器評估策略。

---

## 1. 核心概念

- **價格資料 (`prices`)**：使用 `AggBar` 表示的多標的 OHLCV 資料，欄位至少包含  
  `["start_time", "end_time", "symbol", "open", "high", "low", "close", "volume"]`。
- **信號因子 (`signal`)**：任何 `Factor` 物件，需與 `prices` 在 `end_time, symbol` 上對齊，通常為已做過橫截面處理的排名 / Z-score。
- **`AlphaPipeline`**：將信號轉成可交易權重（正規化 → 配置 → 約束與再正規化）；`Backtester` 預設使用 `RawNormalizer` + `MarketNeutralAllocator`。
- **資產池遮罩 (`mask`)**：可選欄位名稱，用來限制哪些標的在該時間點可以持倉。
- **避免前視偏差**：回測時會使用「前一根 bar 的信號」在當前 bar 交易。
- **向量化實作**：內部全部以 Polars 向量化計算完成，再轉成 pandas 計算績效指標。

主要類別：

- `factorium.backtest.Backtester`：使用者面向的回測器（別名，實際指向 `VectorizedBacktester`）。
- `factorium.backtest.pipeline.AlphaPipeline`：信號到權重的管線（可自訂 normalizer、allocator、constraints）。
- `factorium.backtest.vectorized.VectorizedBacktester`：向量化回測實作。
- `factorium.backtest.vectorized.BacktestResult`：回測結果容器。

---

## 2. 最簡範例：從因子到回測

`Backtester` 的 `pipeline` 參數預設為 `AlphaPipeline()`，等同 **不調整信號（`RawNormalizer`）** 加上 **美元中性權重（`MarketNeutralAllocator`）**。你通常只要指定 `prices`、`signal` 與成本相關參數即可。

```python
from factorium import AggBar
from factorium.backtest import Backtester
import polars as pl

agg = AggBar.from_df(pl.read_parquet("data/multi_symbol.parquet"))

close = agg["close"]
momentum = (close.ts_delta(20) / close.ts_shift(20)).cs_rank()

# 預設：AlphaPipeline() → RawNormalizer + MarketNeutralAllocator
bt = Backtester(
    prices=agg,
    signal=momentum,
    initial_capital=10_000.0,
    transaction_cost=0.0003,   # 可為 float 或 (buy_rate, sell_rate) tuple
    frequency="1h",            # 用於年化指標
)
result = bt.run()

print(result.metrics)
```

若要改為 **只做多** 或套用其他正規化，傳入自訂 `AlphaPipeline`：

```python
from factorium.backtest import AlphaPipeline, LongOnlyAllocator, RankNormalizer

# 橫截面排名到 [0,1] 後再做 long-only 配置
bt = Backtester(
    prices=agg,
    signal=momentum,
    pipeline=AlphaPipeline(
        normalizer=RankNormalizer(),
        allocator=LongOnlyAllocator(),
    ),
    initial_capital=10_000.0,
    transaction_cost=0.0003,
    frequency="1h",
)
result = bt.run()
```

---

## 2.1 AlphaPipeline：三階段與可用元件

`AlphaPipeline` 將每個 `end_time` 橫截面上的信號轉成 `weight`，分三階段：

1. **Normalize（正規化）**：把原始 alpha 映射到可比較的尺度（例如排名、z-score）。
2. **Allocate（配置）**：由正規化後的信號產出權重，並滿足該 allocator 的不變式（例如美元中性或做多總和為 1）。
3. **Constrain + renormalize（約束）**：依序套用 `WeightConstraint`，再由同一個 allocator 的 `renormalize` 恢復不變式。

可由 `factorium.backtest` 匯入的 **Normalizers**：

| 類別 | 說明 |
|------|------|
| `RawNormalizer` | 直通，不更改信號（預設） |
| `RankNormalizer` | 橫截面排名，映射到約 `[0, 1]` |
| `ZScoreNormalizer` | 橫截面 z-score |
| `MinMaxNormalizer` | 橫截面 min-max 到 `[0, 1]`（退化區間時為 null） |

**Allocators**：

| 類別 | 說明 |
|------|------|
| `MarketNeutralAllocator` | 美元中性：`sum(w)=0`、`sum(|w|)=1`（預設） |
| `LongOnlyAllocator` | 只做多：僅 `signal > 0` 分得權重，`sum(w)=1`、`w >= 0` |
| `TopNAllocator(n, long_short=False)` | 每期取信號前 N 名等權；可選多空各 N |

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
    - `var_95`, `cvar_95`
    - `profit_factor`
- **`weights: pl.DataFrame`**  
  - 欄位：`["end_time", "symbol", "weight"]`
  - 每期每標的的最終權重（已套用管線與約束）
- **`turnover: pl.DataFrame`**  
  - 欄位：`["end_time", "turnover", "cost"]`
  - 每期的換手率與交易成本

如需 pandas 版本，可呼叫：

```python
pandas_result = result.to_pandas()
print(pandas_result.equity_curve.tail())
print(pandas_result.metrics)
```

---

## 3.1 使用 Universe / Checklist mask 限制持倉

若你已經在 `AggBar` 內建立 mask 欄位（例如 `checklist_mask`），可以在回測時直接指定。權重仍由 `pipeline` 決定；以下沿用預設市場中性管線，亦可改傳 `AlphaPipeline(allocator=LongOnlyAllocator())` 等。

```python
bt = Backtester(
    prices=agg,
    signal=momentum,
    mask="checklist_mask",
)
result = bt.run()
```

當 `mask` 為 `False` 或 `null`，該標的在該期權重會被設為 0。這能讓回測與因子分析維持同一套 Universe/Checklist 約束。

---

## 4. 市場中性與 Long-only 權重

市場中性與做多模式是由 **`AlphaPipeline` 的 `allocator`** 選擇，而非舊版字串參數。

### 4.1 `MarketNeutralAllocator()`（市場／美元中性）

對正規化後的信號做橫截面去平均，再以 L1 規模正規化：

\[
w_{i,t} = \frac{x_{i,t} - \bar{x}_t}{\sum_j |x_{j,t} - \bar{x}_t|}
\]

其中：

- \(\sum_i w_{i,t} = 0\)
- \(\sum_i |w_{i,t}| = 1\)

實作位於 `factorium.backtest.allocators.MarketNeutralAllocator`；約束套用後會再透過 `renormalize` 維持上述不變式。

### 4.2 `LongOnlyAllocator()`（只做多）

- 僅對 `signal > 0` 的標的配置權重，其餘為 0。
- 每個時間點將正信號加成總和為 1 的長倉組合。

適合與 `RankNormalizer`、`ZScoreNormalizer` 等並用，或在使用已為正的因子時直接搭配 `RawNormalizer`。

---

## 5. 權重約束（Constraints）

約束屬於 **`AlphaPipeline`**，在配置完成後依序套用，並由 allocator **再正規化**。不要將 `constraints` 傳給 `Backtester` 建構子。

- **基底類別**：`factorium.backtest.constraints.WeightConstraint`
- **具體實作**：
  - `MaxPositionConstraint(max_weight: float)`：限制單一標的最大絕對權重。
  - `LongOnlyConstraint()`：將負權重設為 0。
  - `MaxGrossExposureConstraint(max_exposure: float)`：限制每個時間點的總絕對權重。
  - `MarketNeutralConstraint()`：強制每個時間點權重和為 0。

```python
from factorium.backtest import (
    AlphaPipeline,
    Backtester,
    MarketNeutralAllocator,
    MaxPositionConstraint,
)

pipeline = AlphaPipeline(
    allocator=MarketNeutralAllocator(),
    constraints=[
        MaxPositionConstraint(max_weight=0.1),  # 單一標的不超過 10%
    ],
)

bt = Backtester(
    prices=agg,
    signal=momentum,
    pipeline=pipeline,
)
result = bt.run()
```

> **注意**：約束後會再正規化，以恢復 allocator 的不變式（市場中性：`sum(w)=0` 且 `sum(|w|)=1`；long-only：`sum(w)=1` 且 `w>=0`）。正規化可能使個別權重略微超過名義上限，幅度與被截斷權重的佔比有關。

---

## 6. 與 ResearchSession 的整合

實務上可透過 `ResearchSession` 串接資料、因子與回測；`backtest()` 同樣接受 **`pipeline`**（預設為 `AlphaPipeline()`）。

```python
from factorium import ResearchSession
from factorium.backtest import AlphaPipeline, LongOnlyAllocator

session = ResearchSession.from_parquet("data/multi_symbol.parquet")
signal = session.factor("close").ts_delta(20).cs_rank()

result = session.backtest(
    signal,
    pipeline=AlphaPipeline(allocator=LongOnlyAllocator()),
    transaction_cost=0.0003,
)

print(result.metrics)
```

若你需要對不同參數組合進行多個回測，可以在同一個 `ResearchSession` 中重複呼叫 `backtest()`，每次傳入不同的 `pipeline`。

---

## 7. 典型工作流程總結

1. **準備資料**：使用 `BinanceDataLoader.load_aggbar()` 或 `AggBar.from_df()` 建立 `AggBar`。
2. **建立因子**：使用 `AggBar["close"]` 等欄位與 TS/CS 運算子構建 `Factor`。
3. **（可選）分析因子**：使用 `FactorAnalyzer` 或 `ResearchSession.analyze()` 檢查 IC / 分層收益。
4. **執行回測**：
   - 使用 `Backtester(prices=agg, signal=signal, pipeline=...)`（`pipeline` 可省略以使用預設市場中性）；或
   - 透過 `ResearchSession.backtest(signal, pipeline=...)`。
5. **查看結果**：讀取 `BacktestResult.metrics`、`equity_curve`、`weights`、`turnover` 等欄位，或將結果轉成 pandas 作進一步分析。
