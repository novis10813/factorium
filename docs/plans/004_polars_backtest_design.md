# Polars 向量化回測設計

**Status**: Approved  
**Created**: 2026-01-28  
**Based on**: Brainstorming session with user

---

## 1. 設計目標

將現有的迭代式回測引擎重構為向量化 Polars 實作，以提升效能並統一技術棧。

### 設計決策

| 決策點 | 選擇 | 原因 |
|--------|------|------|
| API 模式 | 混合模式 | 內部使用 Polars，但保持 pandas 相容的輸入/輸出 |
| 測試策略 | 遷移到 Polars | 測試也使用 Polars 類型 |
| 執行模式 | Eager (DataFrame) | 相比 LazyFrame，實作較簡單且足夠高效 |
| 計算方式 | 向量化 | 一次計算所有時間點，充分利用 Polars 效能 |
| 現金約束 | 簡化實作 | 先完成基本向量化，之後再加入現金約束 |

---

## 2. 資料模型

### 2.1 核心表格：portfolio_state

```
┌─────────────┬────────┬────────┬──────────────┬─────────────┬───────────────┬─────────────┐
│ end_time    │ symbol │ price  │ signal       │ weight      │ target_qty    │ actual_qty  │
│ i64 (ms)    │ str    │ f64    │ f64 (null ok)│ f64         │ f64           │ f64         │
├─────────────┼────────┼────────┼──────────────┼─────────────┼───────────────┼─────────────┤
│ 1704067200  │ BTC    │ 100.0  │ 0.8          │ 0.25        │ 25.0          │ 25.0        │
│ 1704067200  │ ETH    │ 50.0   │ 0.3          │ -0.25       │ -50.0         │ -50.0       │
└─────────────┴────────┴────────┴──────────────┴─────────────┴───────────────┴─────────────┘
```

### 2.2 彙總表格：equity_history

```
┌─────────────┬───────────┬──────────────┬─────────────┬───────────────┐
│ end_time    │ cash      │ market_value │ total_value │ trade_cost    │
│ i64 (ms)    │ f64       │ f64          │ f64         │ f64           │
├─────────────┼───────────┼──────────────┼─────────────┼───────────────┤
│ 1704067200  │ 8000.0    │ 2000.0       │ 10000.0     │ 0.6           │
└─────────────┴───────────┴──────────────┴─────────────┴───────────────┘
```

---

## 3. 向量化計算流程

### Step 1: 資料準備

```python
# 合併 price 和 signal 資料，使用 shift 避免 lookahead bias
combined = (
    prices_df
    .join(
        signal_df.with_columns(pl.col("end_time").alias("signal_time")),
        left_on=["end_time", "symbol"],
        right_on=["signal_time", "symbol"],
        how="left"
    )
    # signal 來自前一時間點
    .with_columns([
        pl.col("signal").shift(1).over("symbol").alias("prev_signal")
    ])
    .sort(["end_time", "symbol"])
)
```

### Step 2: 權重計算（cross-sectional）

```python
def calculate_weights_vectorized(df: pl.DataFrame, neutralization: str) -> pl.DataFrame:
    if neutralization == "market":
        # Market neutral: (signal - mean) / sum(|signal - mean|)
        return df.with_columns([
            (
                (pl.col("prev_signal") - pl.col("prev_signal").mean().over("end_time"))
                / (pl.col("prev_signal") - pl.col("prev_signal").mean().over("end_time"))
                  .abs().sum().over("end_time")
            ).fill_nan(0.0).alias("weight")
        ])
    else:  # long-only
        # Normalize positive signals to sum to 1
        positive_only = pl.when(pl.col("prev_signal") > 0).then(pl.col("prev_signal")).otherwise(0.0)
        return df.with_columns([
            (positive_only / positive_only.sum().over("end_time")).fill_nan(0.0).alias("weight")
        ])
```

### Step 3: 持倉變化計算

```python
def calculate_positions(df: pl.DataFrame, initial_capital: float) -> pl.DataFrame:
    # 計算每個時間點的總價值（需要累積計算）
    # 簡化版：假設總價值追蹤初始資本
    
    return df.with_columns([
        # 目標持倉 = 權重 * 總價值 / 價格
        (pl.col("weight") * initial_capital / pl.col("price")).alias("target_qty"),
        # 前一持倉
        pl.col("target_qty").shift(1).over("symbol").fill_null(0.0).alias("prev_qty"),
    ]).with_columns([
        # 交易量
        (pl.col("target_qty") - pl.col("prev_qty")).alias("trade_qty")
    ])
```

### Step 4: 權益計算

```python
def calculate_equity(df: pl.DataFrame, initial_capital: float, cost_rate: float) -> pl.DataFrame:
    # 彙總每個時間點
    equity = (
        df.group_by("end_time")
        .agg([
            # 市值
            (pl.col("target_qty") * pl.col("price")).sum().alias("market_value"),
            # 交易成本
            (pl.col("trade_qty").abs() * pl.col("price") * cost_rate).sum().alias("trade_cost"),
            # 淨買入金額（正=買入，負=賣出）
            (pl.col("trade_qty") * pl.col("price")).sum().alias("net_buy"),
        ])
        .sort("end_time")
    )
    
    # 累積計算現金
    return equity.with_columns([
        (initial_capital 
         - pl.col("trade_cost").cum_sum() 
         - pl.col("net_buy").cum_sum()
        ).alias("cash"),
    ]).with_columns([
        (pl.col("cash") + pl.col("market_value")).alias("total_value")
    ])
```

---

## 4. API 設計

### 4.1 VectorizedBacktester

```python
from factorium.backtest import VectorizedBacktester, BacktestResult

bt = VectorizedBacktester(
    prices: pl.DataFrame | AggBar,
    signal: pl.DataFrame | Factor,
    initial_capital: float = 10000.0,
    transaction_cost: float | tuple[float, float] = 0.0003,
    neutralization: Literal["market", "none"] = "market",
    constraint: WeightConstraint | None = None,
    frequency: str = "1h",
)

result: BacktestResult = bt.run()
```

### 4.2 BacktestResult（更新）

```python
@dataclass
class BacktestResult:
    equity_curve: pl.DataFrame      # columns: [end_time, total_value]
    returns: pl.DataFrame           # columns: [end_time, return]
    metrics: Dict[str, float]
    trades: pl.DataFrame            # columns: [end_time, symbol, qty, price, cost]
    portfolio_history: pl.DataFrame # columns: [end_time, cash, market_value, total_value]
    
    def to_pandas(self) -> "BacktestResultPandas":
        """Convert all DataFrames to pandas for compatibility."""
        ...
```

### 4.3 向後相容

```python
# 舊的 Backtester 作為 alias
class Backtester(VectorizedBacktester):
    """Legacy alias for VectorizedBacktester."""
    pass
```

---

## 5. 實作計畫整合

此設計將整合到 `docs/plans/003_research_engine_implementation.md`：

- **Phase 0** 更新為：Polars 向量化回測重構
- **Phase B/C** 保持不變，但使用新的 Polars API

### 新增任務

1. **Task 0.1**: 建立向量化回測核心引擎
2. **Task 0.2**: 實作權重計算（market neutral / long-only）
3. **Task 0.3**: 實作持倉和交易計算
4. **Task 0.4**: 實作權益追蹤和指標計算
5. **Task 0.5**: 向後相容層和測試遷移

---

## 6. 測試策略

### 單元測試

```python
def test_weights_calculation_market_neutral():
    """Cross-sectional weights should be dollar-neutral."""
    signals = pl.DataFrame({
        "end_time": [1, 1, 1, 1],
        "symbol": ["A", "B", "C", "D"],
        "signal": [0.8, 0.5, 0.3, 0.1],
    })
    weights = calculate_weights_vectorized(signals, "market")
    
    # Weights should sum to 0
    assert abs(weights["weight"].sum()) < 1e-10
    # Absolute weights should sum to 1
    assert abs(weights["weight"].abs().sum() - 1.0) < 1e-10
```

### 整合測試

```python
def test_vectorized_backtester_produces_valid_equity():
    """Equity curve should be monotonically bounded."""
    bt = VectorizedBacktester(prices=sample_prices, signal=sample_signal)
    result = bt.run()
    
    # Total value should never be negative
    assert result.equity_curve["total_value"].min() > 0
    # Returns should be finite
    assert result.returns["return"].is_finite().all()
```

### 一致性測試

```python
def test_vectorized_matches_iterative():
    """Vectorized and iterative implementations should produce similar results."""
    # Compare with legacy iterative implementation
    # Allow small numerical differences
    ...
```

---

## 7. 效能預期

| 資料規模 | 迭代式 (估計) | 向量化 (預期) |
|----------|---------------|---------------|
| 1K bars  | ~100ms        | ~10ms         |
| 10K bars | ~1s           | ~50ms         |
| 100K bars| ~10s          | ~200ms        |

向量化實作預期可提升 10-50 倍效能。

---

## 8. 風險與限制

1. **現金約束簡化**：初版不會實作嚴格的現金不足拒絕，會在未來版本加入
2. **數值精度**：向量化計算可能與迭代式有微小差異，需要設定容忍度
3. **記憶體使用**：一次載入所有資料，大型資料集需注意記憶體

---

## 9. 下一步

1. 更新 `003_research_engine_implementation.md` 計畫
2. 開始實作 VectorizedBacktester
3. 遷移現有測試到 Polars
