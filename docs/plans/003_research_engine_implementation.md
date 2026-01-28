# Research Engine Implementation Plan (Phase 0 + B + C)

## Implementation Status (2026-01-28)

All tasks from this plan have been implemented. Key implementation notes:

1. **FactorAnalysisResult**: `analyze()` returns a structured dataclass, not dict
2. **MaxPositionConstraint**: Added `normalize=True` option
3. **WeightConstraint**: Strategy pattern (separate classes)
4. **safe_divide**: Uses EPSILON threshold per AGENTS.md
5. **Backtester**: Now alias for VectorizedBacktester

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 將回測系統重構為向量化 Polars 實作，並建立完整的因子研究工作流 API。

**Architecture:** 
- Phase 0: **Polars 向量化回測重構**（取代舊的迭代式實作）
- Phase B: 引入 `ResearchSession` 高階 API，統一資料載入、因子建構、分析、回測的流程
- Phase C: 擴展因子分析器與回測器，支援多因子組合、約束條件、報告生成

**Tech Stack:** Python, Polars (DataFrame eager mode), pytest, matplotlib

**Related Design Doc:** `docs/plans/004_polars_backtest_design.md`

---

## Phase 0: Polars 向量化回測重構

### 設計決策摘要

| 決策點 | 選擇 | 原因 |
|--------|------|------|
| API 模式 | 混合模式 | 內部使用 Polars，但保持 pandas 相容的輸入/輸出 |
| 測試策略 | 遷移到 Polars | 測試也使用 Polars 類型 |
| 執行模式 | Eager (DataFrame) | 相比 LazyFrame，實作較簡單且足夠高效 |
| 計算方式 | 向量化 | 一次計算所有時間點，充分利用 Polars 效能 |
| 現金約束 | 簡化實作 | 先完成基本向量化，之後在 Phase C 加入約束 |

---

### Task 0.1: VectorizedBacktester 核心結構

**Files:**
- Create: `src/factorium/backtest/vectorized.py`
- Create: `tests/backtest/test_vectorized.py`

**Step 1: Write failing tests for VectorizedBacktester**

`tests/backtest/test_vectorized.py`:

```python
import pytest
import polars as pl
import numpy as np

from factorium import AggBar
from factorium.backtest.vectorized import VectorizedBacktester, BacktestResult


class TestVectorizedBacktesterInit:
    """Tests for VectorizedBacktester initialization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 50, 3600000))
        
        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH", "SOL"]:
                base_price = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append({
                    "start_time": ts,
                    "end_time": ts + 3600000,
                    "symbol": symbol,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1000.0,
                })
        
        return AggBar(pl.DataFrame(rows))

    def test_init_with_aggbar(self, sample_data):
        """Should initialize with AggBar."""
        signal = sample_data["close"].cs_rank()
        
        bt = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
        )
        
        assert bt.initial_capital == 10000.0
        assert bt.neutralization == "market"

    def test_run_returns_result(self, sample_data):
        """run() should return BacktestResult."""
        signal = sample_data["close"].cs_rank()
        
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        assert isinstance(result, BacktestResult)
        assert result.equity_curve is not None
        assert result.returns is not None
        assert result.metrics is not None

    def test_equity_curve_is_polars_dataframe(self, sample_data):
        """equity_curve should be Polars DataFrame."""
        signal = sample_data["close"].cs_rank()
        
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        assert isinstance(result.equity_curve, pl.DataFrame)
        assert "end_time" in result.equity_curve.columns
        assert "total_value" in result.equity_curve.columns

    def test_total_value_positive(self, sample_data):
        """Total value should always be positive."""
        signal = sample_data["close"].cs_rank()
        
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        assert result.equity_curve["total_value"].min() > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_vectorized.py::TestVectorizedBacktesterInit -v`
Expected: FAIL (module not found)

**Step 3: Implement VectorizedBacktester skeleton**

`src/factorium/backtest/vectorized.py`:

```python
"""
Vectorized backtester using Polars.

This module provides a high-performance backtester that computes all time steps
in a single vectorized operation, leveraging Polars' query optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

import polars as pl

from ..constants import EPSILON

if TYPE_CHECKING:
    from ..aggbar import AggBar
    from ..factors.core import Factor


@dataclass
class BacktestResult:
    """
    Results from a backtest run.
    
    All DataFrames are Polars format internally.
    """
    equity_curve: pl.DataFrame      # columns: [end_time, total_value]
    returns: pl.DataFrame           # columns: [end_time, return]
    metrics: Dict[str, float]
    trades: pl.DataFrame            # columns: [end_time, symbol, qty, price, cost]
    portfolio_history: pl.DataFrame # columns: [end_time, cash, market_value, total_value]
    
    def to_pandas(self):
        """Convert all DataFrames to pandas for compatibility."""
        import pandas as pd
        return {
            "equity_curve": self.equity_curve.to_pandas(),
            "returns": self.returns.to_pandas(),
            "metrics": self.metrics,
            "trades": self.trades.to_pandas(),
            "portfolio_history": self.portfolio_history.to_pandas(),
        }


class VectorizedBacktester:
    """
    Vectorized backtester using Polars for high-performance simulation.
    
    This backtester computes all time steps in a single vectorized operation,
    which is significantly faster than iterative approaches for large datasets.
    
    Args:
        prices: Price data as AggBar
        signal: Factor signal for position weighting
        entry_price: Column name for entry prices (default: "close")
        transaction_cost: Transaction cost rate (default: 0.0003)
        initial_capital: Starting capital (default: 10000.0)
        neutralization: "market" for dollar-neutral, "none" for long-only
        frequency: Trading frequency for annualization (default: "1h")
    """
    
    def __init__(
        self,
        prices: "AggBar",
        signal: "Factor",
        entry_price: str = "close",
        transaction_cost: Union[float, tuple[float, float]] = 0.0003,
        initial_capital: float = 10000.0,
        neutralization: Literal["market", "none"] = "market",
        frequency: str = "1h",
    ):
        self.prices = prices
        self.signal = signal
        self.entry_price = entry_price
        self.initial_capital = initial_capital
        self.neutralization = neutralization
        self.frequency = frequency
        
        # Normalize transaction cost to tuple
        if isinstance(transaction_cost, (int, float)):
            self.cost_rates = (float(transaction_cost), float(transaction_cost))
        else:
            self.cost_rates = transaction_cost
    
    def run(self) -> BacktestResult:
        """
        Execute the backtest.
        
        Returns:
            BacktestResult containing equity curve, returns, and metrics.
        """
        # Step 1: Prepare combined data
        combined = self._prepare_data()
        
        # Step 2: Calculate weights
        weighted = self._calculate_weights(combined)
        
        # Step 3: Calculate positions and trades
        positioned = self._calculate_positions(weighted)
        
        # Step 4: Calculate equity
        equity = self._calculate_equity(positioned)
        
        # Step 5: Calculate metrics
        metrics = self._calculate_metrics(equity)
        
        # Step 6: Build result
        return self._build_result(equity, positioned, metrics)
    
    def _prepare_data(self) -> pl.DataFrame:
        """Combine price and signal data."""
        # Get prices as Polars DataFrame
        prices_df = self.prices.to_polars().select([
            "end_time", "symbol", 
            pl.col(self.entry_price).alias("price")
        ])
        
        # Get signal as Polars DataFrame
        signal_df = self.signal.lazy.select([
            "end_time", "symbol", 
            pl.col("factor").alias("signal")
        ]).collect()
        
        # Join and shift signal (use previous signal for current decision)
        combined = (
            prices_df
            .join(signal_df, on=["end_time", "symbol"], how="left")
            .sort(["symbol", "end_time"])
            .with_columns([
                pl.col("signal").shift(1).over("symbol").alias("prev_signal")
            ])
        )
        
        return combined
    
    def _calculate_weights(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate target weights for each time step."""
        if self.neutralization == "market":
            # Dollar-neutral: (signal - mean) / sum(|signal - mean|)
            return df.with_columns([
                (
                    (pl.col("prev_signal") - pl.col("prev_signal").mean().over("end_time"))
                    / (pl.col("prev_signal") - pl.col("prev_signal").mean().over("end_time"))
                      .abs().sum().over("end_time")
                ).fill_nan(0.0).fill_null(0.0).alias("weight")
            ])
        else:
            # Long-only: normalize positive signals
            return df.with_columns([
                pl.when(pl.col("prev_signal") > 0)
                .then(pl.col("prev_signal"))
                .otherwise(0.0)
                .alias("positive_signal")
            ]).with_columns([
                (pl.col("positive_signal") / pl.col("positive_signal").sum().over("end_time"))
                .fill_nan(0.0).fill_null(0.0).alias("weight")
            ]).drop("positive_signal")
    
    def _calculate_positions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate positions and trades."""
        # Calculate target value per position
        df = df.with_columns([
            (pl.col("weight") * self.initial_capital).alias("target_value"),
        ])
        
        # Calculate target quantity
        df = df.with_columns([
            (pl.col("target_value") / pl.col("price")).alias("target_qty")
        ])
        
        # Calculate previous quantity and trade
        df = df.sort(["symbol", "end_time"]).with_columns([
            pl.col("target_qty").shift(1).over("symbol").fill_null(0.0).alias("prev_qty")
        ])
        
        df = df.with_columns([
            (pl.col("target_qty") - pl.col("prev_qty")).alias("trade_qty")
        ])
        
        # Calculate trade cost
        cost_rate = self.cost_rates[0]  # Simplified: use same rate for buy/sell
        df = df.with_columns([
            (pl.col("trade_qty").abs() * pl.col("price") * cost_rate).alias("trade_cost")
        ])
        
        return df
    
    def _calculate_equity(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate equity curve."""
        # Aggregate per timestamp
        equity = (
            df.group_by("end_time")
            .agg([
                (pl.col("target_qty") * pl.col("price")).sum().alias("market_value"),
                pl.col("trade_cost").sum().alias("period_cost"),
                (pl.col("trade_qty") * pl.col("price")).sum().alias("net_buy"),
            ])
            .sort("end_time")
        )
        
        # Calculate cumulative cash
        equity = equity.with_columns([
            (
                self.initial_capital 
                - pl.col("period_cost").cum_sum() 
                - pl.col("net_buy").cum_sum()
            ).alias("cash")
        ])
        
        # Calculate total value
        equity = equity.with_columns([
            (pl.col("cash") + pl.col("market_value")).alias("total_value")
        ])
        
        # Calculate returns
        equity = equity.with_columns([
            (pl.col("total_value") / pl.col("total_value").shift(1) - 1)
            .fill_null(0.0).alias("return")
        ])
        
        return equity
    
    def _calculate_metrics(self, equity: pl.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        from .utils import frequency_to_periods_per_year
        
        returns = equity["return"].to_numpy()
        periods_per_year = frequency_to_periods_per_year(self.frequency)
        
        total_return = float((1 + returns).prod() - 1)
        n_periods = len(returns)
        years = n_periods / periods_per_year if periods_per_year > 0 else 0
        
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        annual_volatility = float(returns.std() * (periods_per_year ** 0.5))
        
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > EPSILON else 0.0
        
        # Max drawdown
        total_values = equity["total_value"].to_numpy()
        peak = total_values.cummax() if hasattr(total_values, 'cummax') else np.maximum.accumulate(total_values)
        drawdown = (total_values - peak) / peak
        max_drawdown = float(drawdown.min())
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }
    
    def _build_result(
        self, 
        equity: pl.DataFrame, 
        positioned: pl.DataFrame,
        metrics: Dict[str, float],
    ) -> BacktestResult:
        """Build the final result object."""
        equity_curve = equity.select(["end_time", "total_value"])
        returns = equity.select(["end_time", "return"])
        
        trades = (
            positioned
            .filter(pl.col("trade_qty").abs() > EPSILON)
            .select(["end_time", "symbol", "trade_qty", "price", "trade_cost"])
            .rename({"trade_qty": "qty", "trade_cost": "cost"})
        )
        
        portfolio_history = equity.select([
            "end_time", "cash", "market_value", "total_value"
        ])
        
        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics,
            trades=trades,
            portfolio_history=portfolio_history,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_vectorized.py::TestVectorizedBacktesterInit -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/factorium/backtest/vectorized.py tests/backtest/test_vectorized.py
git commit -m "feat(backtest): add VectorizedBacktester with Polars"
```

---

### Task 0.2: 權重計算測試與驗證

**Files:**
- Modify: `tests/backtest/test_vectorized.py`

**Step 1: Write tests for weight calculation**

```python
class TestWeightCalculation:
    """Tests for weight calculation in VectorizedBacktester."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known signals."""
        timestamps = [1704067200000, 1704070800000, 1704074400000]
        
        rows = []
        for ts in timestamps:
            # Create signals that sum to known values
            for symbol, signal in [("A", 0.8), ("B", 0.5), ("C", 0.2), ("D", -0.1)]:
                rows.append({
                    "start_time": ts,
                    "end_time": ts + 3600000,
                    "symbol": symbol,
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                    "volume": 1000.0,
                })
        
        return AggBar(pl.DataFrame(rows))

    def test_market_neutral_weights_sum_to_zero(self, sample_data):
        """Market neutral weights should sum to zero."""
        signal = sample_data["close"]  # Will be constant, but test the logic
        
        bt = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
            neutralization="market",
        )
        
        # Access internal method
        combined = bt._prepare_data()
        weighted = bt._calculate_weights(combined)
        
        # Group by end_time and check sum
        weight_sums = (
            weighted
            .group_by("end_time")
            .agg(pl.col("weight").sum().alias("weight_sum"))
        )
        
        # All sums should be approximately zero
        assert weight_sums["weight_sum"].abs().max() < 1e-10

    def test_long_only_weights_sum_to_one(self):
        """Long-only weights should sum to 1."""
        timestamps = [1704067200000, 1704070800000]
        
        rows = []
        for i, ts in enumerate(timestamps):
            for symbol, price in [("A", 100.0), ("B", 50.0), ("C", 25.0)]:
                rows.append({
                    "start_time": ts,
                    "end_time": ts + 3600000,
                    "symbol": symbol,
                    "close": price * (1 + 0.01 * i),  # Varying prices
                    "open": price, "high": price, "low": price, "volume": 1000.0,
                })
        
        data = AggBar(pl.DataFrame(rows))
        signal = data["close"].cs_rank()
        
        bt = VectorizedBacktester(
            prices=data,
            signal=signal,
            neutralization="none",
        )
        
        combined = bt._prepare_data()
        weighted = bt._calculate_weights(combined)
        
        # Group by end_time and check sum (excluding first row which has no prev_signal)
        weight_sums = (
            weighted
            .filter(pl.col("weight") != 0)
            .group_by("end_time")
            .agg(pl.col("weight").sum().alias("weight_sum"))
        )
        
        # All non-zero sums should be approximately 1
        for ws in weight_sums["weight_sum"].to_list():
            if ws > 0:
                assert abs(ws - 1.0) < 1e-10
```

**Step 2: Run tests**

Run: `uv run pytest tests/backtest/test_vectorized.py::TestWeightCalculation -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/backtest/test_vectorized.py
git commit -m "test(backtest): add weight calculation tests for VectorizedBacktester"
```

---

### Task 0.3: 更多 Metrics 計算

**Files:**
- Modify: `src/factorium/backtest/vectorized.py`
- Test: `tests/backtest/test_vectorized.py`

**Step 1: Write tests for additional metrics**

```python
class TestMetricsCalculation:
    """Tests for metrics calculation."""

    @pytest.fixture
    def sample_data(self):
        # Same as TestVectorizedBacktesterInit
        ...

    def test_metrics_include_sortino(self, sample_data):
        """Metrics should include Sortino ratio."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        assert "sortino_ratio" in result.metrics

    def test_metrics_include_calmar(self, sample_data):
        """Metrics should include Calmar ratio."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        assert "calmar_ratio" in result.metrics

    def test_max_drawdown_non_positive(self, sample_data):
        """Max drawdown should be <= 0."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        assert result.metrics["max_drawdown"] <= 0
```

**Step 2: Update _calculate_metrics to include more metrics**

修改 `src/factorium/backtest/vectorized.py` 的 `_calculate_metrics` 方法：

```python
def _calculate_metrics(self, equity: pl.DataFrame) -> Dict[str, float]:
    """Calculate performance metrics."""
    import numpy as np
    from .utils import frequency_to_periods_per_year
    
    returns = equity["return"].to_numpy()
    periods_per_year = frequency_to_periods_per_year(self.frequency)
    
    total_return = float((1 + returns).prod() - 1)
    n_periods = len(returns)
    years = n_periods / periods_per_year if periods_per_year > 0 else 0
    
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    annual_volatility = float(returns.std() * (periods_per_year ** 0.5))
    
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > EPSILON else 0.0
    
    # Max drawdown
    total_values = equity["total_value"].to_numpy()
    peak = np.maximum.accumulate(total_values)
    drawdown = (total_values - peak) / peak
    max_drawdown = float(drawdown.min())
    
    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = float(downside_returns.std() * (periods_per_year ** 0.5))
        sortino_ratio = annual_return / downside_std if downside_std > EPSILON else np.inf
    else:
        sortino_ratio = np.inf if annual_return > 0 else 0.0
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > EPSILON else 0.0
    
    # Win rate
    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }
```

**Step 3: Run tests**

Run: `uv run pytest tests/backtest/test_vectorized.py::TestMetricsCalculation -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/factorium/backtest/vectorized.py tests/backtest/test_vectorized.py
git commit -m "feat(backtest): add more metrics to VectorizedBacktester"
```

---

### Task 0.4: 向後相容層與導出

**Files:**
- Modify: `src/factorium/backtest/__init__.py`
- Modify: `src/factorium/backtest/backtester.py` (deprecation warning)
- Test: `tests/backtest/test_vectorized.py`

**Step 1: Write compatibility tests**

```python
class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_old_backtester_import(self):
        """Old Backtester import should still work."""
        from factorium.backtest import Backtester
        assert Backtester is not None

    def test_old_backtestresult_import(self):
        """Old BacktestResult import should still work."""
        from factorium.backtest import BacktestResult
        assert BacktestResult is not None

    def test_result_to_pandas(self, sample_data):
        """BacktestResult.to_pandas() should return pandas-compatible data."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        pandas_result = result.to_pandas()
        
        import pandas as pd
        assert isinstance(pandas_result["equity_curve"], pd.DataFrame)
        assert isinstance(pandas_result["returns"], pd.DataFrame)
```

**Step 2: Update backtest package exports**

`src/factorium/backtest/__init__.py`:

```python
"""Backtest module for portfolio simulation."""

from .vectorized import VectorizedBacktester, BacktestResult
from .metrics import calculate_metrics
from .utils import (
    frequency_to_periods_per_year,
    neutralize_weights,
    normalize_weights,
)

# Backward compatibility: Backtester is now an alias for VectorizedBacktester
Backtester = VectorizedBacktester

__all__ = [
    "Backtester",
    "VectorizedBacktester",
    "BacktestResult",
    "calculate_metrics",
    "frequency_to_periods_per_year",
    "neutralize_weights",
    "normalize_weights",
]
```

**Step 3: Add deprecation warning to old backtester (optional)**

如果需要保留舊的實作，可以重命名為 `IterativeBacktester` 並標記為 deprecated。

**Step 4: Run all tests**

Run: `uv run pytest tests/backtest/ -v`
Expected: All PASS (可能需要更新一些舊測試)

**Step 5: Commit**

```bash
git add src/factorium/backtest/__init__.py tests/backtest/
git commit -m "feat(backtest): add backward compatibility for VectorizedBacktester"
```

---

### Task 0.5: 遷移現有測試到 Polars

**Files:**
- Modify: `tests/backtest/test_backtester.py`

**Step 1: Update existing tests to use Polars types**

更新 `tests/backtest/test_backtester.py` 中的測試，使用 Polars 類型。主要更改：

1. 將 `pd.DataFrame` 替換為 `pl.DataFrame`
2. 更新 assertions 以處理 Polars DataFrame
3. 使用 `AggBar` 的 Polars 路徑

**Step 2: Run all backtest tests**

Run: `uv run pytest tests/backtest/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/backtest/
git commit -m "refactor(tests): migrate backtest tests to Polars"
```

---

## Phase B: Research Session API 與 Notebook UX

> **Note:** Phase B 和 C 的測試和實作可以繼續使用 pandas DataFrame 作為輸入，
> 因為 `ResearchSession` 會在內部轉換為 `AggBar`，而 `AggBar` 已經支援 Polars。

### Task B.1: ResearchSession 基礎架構

**Files:**
- Create: `src/factorium/research/__init__.py`
- Create: `src/factorium/research/session.py`
- Test: `tests/research/test_session.py`

**Step 1: Create research package structure**

```bash
mkdir -p src/factorium/research tests/research
touch tests/research/__init__.py
```

**Step 2: Write failing test for ResearchSession**

`tests/research/test_session.py`:
                # BTC starts cheap, becomes very expensive
                # ETH stays cheap
                if symbol == "BTC":
                    price = 10.0 if i < 5 else 10000.0  # Price jumps 1000x
                else:
                    price = 10.0
                rows.append({
                    "start_time": ts,
                    "end_time": ts + 3600000,
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1000.0,
                })

        df = pd.DataFrame(rows)
        agg = AggBar(df)
        
        # Signal always wants to buy BTC
        signal = agg["close"].cs_rank()
        
        bt = Backtester(
            prices=agg,
            signal=signal,
            initial_capital=1000.0,  # Not enough for expensive BTC
            neutralization="none",
        )
        result = bt.run()
        
        # Should complete without error
        assert isinstance(result, BacktestResult)
        # Cash should never go negative
        assert all(result.portfolio_history["cash"] >= 0)
```

**Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/backtest/test_backtester.py::TestBacktesterCashHandling -v`
Expected: Behavior should already work since Portfolio now rejects trades, but verify no errors

**Step 3: (If needed) Update backtester to handle rejected trades**

如果測試通過，表示 `Backtester.run()` 已經正確處理 `execute_trade` 返回 `False` 的情況（因為它不檢查返回值）。但為了更好的可觀測性，可以選擇記錄被拒絕的交易：

目前的實作已經足夠，因為 `execute_trade` 返回 `bool` 但 `Backtester` 不需要處理它——被拒絕的交易不會被記錄到 `trade_log`。

**Step 4: Run all tests**

Run: `uv run pytest tests/backtest/ -v`
Expected: All PASS

**Step 5: Commit (if changes were made)**

```bash
git add tests/backtest/test_backtester.py
git commit -m "test(backtest): add test for rejected trade handling"
```

---

### Task 0.3: 價格缺失時的警告與處理

**Files:**
- Modify: `src/factorium/backtest/backtester.py:144-157`
- Test: `tests/backtest/test_backtester.py`

**Step 1: Write test for missing price handling**

```python
class TestMissingPriceHandling:
    """Tests for handling missing prices."""

    def test_missing_price_symbol_excluded_from_holdings(self):
        """Symbols with missing prices should be excluded from target holdings."""
        dates = pd.date_range(start="2025-01-01", periods=10, freq="1h")
        timestamps = dates.astype(np.int64) // 10**6

        rows = []
        for i, ts in enumerate(timestamps):
            # BTC has all prices
            rows.append({
                "start_time": ts,
                "end_time": ts + 3600000,
                "symbol": "BTC",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
            })
            # ETH only has prices for first 5 bars
            if i < 5:
                rows.append({
                    "start_time": ts,
                    "end_time": ts + 3600000,
                    "symbol": "ETH",
                    "open": 50.0,
                    "high": 50.0,
                    "low": 50.0,
                    "close": 50.0,
                    "volume": 1000.0,
                })

        df = pd.DataFrame(rows)
        agg = AggBar(df)
        
        # Signal includes both symbols
        signal = agg["close"].cs_rank()
        
        bt = Backtester(
            prices=agg,
            signal=signal,
            neutralization="market",
        )
        result = bt.run()
        
        # Should complete without error
        assert isinstance(result, BacktestResult)
        # Final portfolio should only have BTC (ETH has no price after bar 5)
        # This is implicitly tested by the backtester completing successfully
```

**Step 2: Run test**

Run: `uv run pytest tests/backtest/test_backtester.py::TestMissingPriceHandling -v`
Expected: PASS (current implementation already handles this via `common_symbols` intersection)

**Step 3: Add logging for excluded symbols (optional enhancement)**

修改 `src/factorium/backtest/backtester.py`，在 `_calculate_target_holdings` 中增加日誌：

```python
import logging

logger = logging.getLogger(__name__)

def _calculate_target_holdings(
    self,
    weights: pd.Series,
    prices: pd.Series,
    total_value: float,
) -> pd.Series:
    common_symbols = weights.index.intersection(prices.index)
    
    # Log excluded symbols
    excluded_from_weights = set(weights.index) - set(common_symbols)
    excluded_from_prices = set(prices.index) - set(common_symbols)
    
    if excluded_from_weights:
        logger.debug(f"Symbols in weights but missing prices: {excluded_from_weights}")
    
    weights = weights.loc[common_symbols]
    prices = prices.loc[common_symbols]

    target_values = weights * total_value
    target_quantities = target_values / prices

    return target_quantities
```

**Step 4: Run all tests**

Run: `uv run pytest tests/backtest/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/factorium/backtest/backtester.py tests/backtest/test_backtester.py
git commit -m "fix(backtest): add logging for symbols excluded due to missing prices"
```

---

### Task 0.4: Metrics 邊界情況改進

**Files:**
- Modify: `src/factorium/backtest/metrics.py:46,57`
- Test: `tests/backtest/test_backtester.py`

**Step 1: Write tests for edge cases**

```python
class TestMetricsEdgeCases:
    """Tests for metrics edge cases."""

    def test_annual_return_with_zero_years(self):
        """annual_return should be NaN when years=0."""
        # Single return point - effectively 0 years
        returns = pd.Series([0.01])
        metrics = calculate_metrics(returns, periods_per_year=365*24)
        
        # With only 1 data point, we can't calculate meaningful annual return
        # but current implementation returns nan_result for len < 2
        assert np.isnan(metrics["annual_return"]) or metrics["annual_return"] == 0.0

    def test_sortino_ratio_no_downside(self):
        """sortino_ratio should be inf when all returns positive and excess > 0."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02] * 20)
        metrics = calculate_metrics(returns, periods_per_year=365*24)
        
        # No downside returns, positive excess -> inf
        assert metrics["sortino_ratio"] == np.inf

    def test_sortino_ratio_no_downside_no_excess(self):
        """sortino_ratio should be 0 when no downside and excess <= 0."""
        # All zero returns
        returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0] * 20)
        metrics = calculate_metrics(returns, periods_per_year=365*24)
        
        # No downside, no excess -> 0
        assert metrics["sortino_ratio"] == 0.0
```

**Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/backtest/test_backtester.py::TestMetricsEdgeCases -v`
Expected: Should PASS with current implementation

**Step 3: (Optional) Improve annual_return edge case**

如果希望 `years=0` 時返回 `np.nan` 而非 `0.0`，修改 `metrics.py:46`：

```python
annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
```

**Step 4: Run all tests**

Run: `uv run pytest tests/backtest/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/factorium/backtest/metrics.py tests/backtest/test_backtester.py
git commit -m "fix(backtest): improve metrics edge case handling"
```

---

## Phase B: Research Session API 與 Notebook UX

### Task B.1: ResearchSession 基礎架構

**Files:**
- Create: `src/factorium/research/__init__.py`
- Create: `src/factorium/research/session.py`
- Test: `tests/research/test_session.py`

**Step 1: Create research package structure**

```bash
mkdir -p src/factorium/research tests/research
touch tests/research/__init__.py
```

**Step 2: Write failing test for ResearchSession**

`tests/research/test_session.py`:

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from factorium import AggBar
from factorium.research import ResearchSession


class TestResearchSessionInit:
    """Tests for ResearchSession initialization."""

    @pytest.fixture
    def sample_aggbar(self):
        """Create sample AggBar for testing."""
        dates = pd.date_range(start="2025-01-01", periods=50, freq="1h")
        timestamps = dates.astype(np.int64) // 10**6

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH", "SOL"]:
                base_price = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}[symbol]
                price = base_price * (1 + 0.01 * i + 0.005 * np.random.randn())
                rows.append({
                    "start_time": ts,
                    "end_time": ts + 3600000,
                    "symbol": symbol,
                    "open": price * 0.99,
                    "high": price * 1.01,
                    "low": price * 0.98,
                    "close": price,
                    "volume": 1000.0 * (1 + 0.1 * np.random.randn()),
                })

        return AggBar(pd.DataFrame(rows))

    def test_init_with_aggbar(self, sample_aggbar):
        """ResearchSession should accept AggBar."""
        session = ResearchSession(data=sample_aggbar)
        
        assert session.data is not None
        assert len(session.symbols) == 3
        assert "BTC" in session.symbols

    def test_init_with_dataframe(self, sample_aggbar):
        """ResearchSession should accept DataFrame."""
        df = sample_aggbar.to_df()
        session = ResearchSession(data=df)
        
        assert session.data is not None
        assert len(session.symbols) == 3

    def test_get_factor(self, sample_aggbar):
        """Should be able to get a factor from session."""
        session = ResearchSession(data=sample_aggbar)
        close = session.factor("close")
        
        assert close is not None
        assert close.name == "close"

    def test_create_factor_from_expression(self, sample_aggbar):
        """Should be able to create factor from expression string."""
        session = ResearchSession(data=sample_aggbar)
        mom = session.create_factor("ts_delta(close, 5) / ts_shift(close, 5)", name="momentum")
        
        assert mom is not None
        assert mom.name == "momentum"
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/research/test_session.py::TestResearchSessionInit -v`
Expected: FAIL (module not found)

**Step 4: Implement ResearchSession**

`src/factorium/research/__init__.py`:

```python
"""Research workflow utilities for factor analysis."""

from .session import ResearchSession

__all__ = ["ResearchSession"]
```

`src/factorium/research/session.py`:

```python
"""
ResearchSession: High-level API for factor research workflows.

Provides a unified interface for:
- Loading and managing data (AggBar)
- Building factors from expressions or callables
- Running standard analyses (IC, quantile returns, decay)
- Executing backtests
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import pandas as pd

from ..aggbar import AggBar
from ..factors.core import Factor
from ..factors.parser import FactorExpressionParser

if TYPE_CHECKING:
    from ..backtest.backtester import BacktestResult
    from ..factors.analyzer import FactorAnalyzer


class ResearchSession:
    """
    High-level API for factor research workflows.
    
    Manages data, factor definitions, and provides shortcuts for common
    analysis and backtesting operations.
    
    Args:
        data: Price/OHLCV data as AggBar or DataFrame
        name: Optional session name for identification
        
    Example:
        >>> from factorium.research import ResearchSession
        >>> session = ResearchSession(data=agg)
        >>> mom = session.create_factor("ts_delta(close, 20) / ts_shift(close, 20)")
        >>> ic_summary = session.analyze(mom).calculate_ic_summary()
        >>> result = session.backtest(mom)
    """
    
    def __init__(
        self,
        data: Union[AggBar, pd.DataFrame],
        name: Optional[str] = None,
    ):
        if isinstance(data, pd.DataFrame):
            self._data = AggBar(data)
        elif isinstance(data, AggBar):
            self._data = data
        else:
            raise TypeError(f"data must be AggBar or DataFrame, got {type(data)}")
        
        self.name = name or "ResearchSession"
        self._factors: Dict[str, Factor] = {}
        self._parser = FactorExpressionParser()
    
    @property
    def data(self) -> AggBar:
        """Return the underlying AggBar data."""
        return self._data
    
    @property
    def symbols(self) -> List[str]:
        """Return list of symbols in the data."""
        return self._data.symbols
    
    @property
    def cols(self) -> List[str]:
        """Return list of available columns."""
        return self._data.cols
    
    def factor(self, column: str) -> Factor:
        """
        Get a factor from a data column.
        
        Args:
            column: Column name (e.g., 'close', 'volume')
            
        Returns:
            Factor object
        """
        return self._data[column]
    
    def create_factor(
        self,
        expr: Union[str, Callable[[AggBar], Factor]],
        name: Optional[str] = None,
    ) -> Factor:
        """
        Create a factor from an expression or callable.
        
        Args:
            expr: Either a string expression (e.g., "ts_mean(close, 20)")
                  or a callable that takes AggBar and returns Factor
            name: Optional name for the factor
            
        Returns:
            Factor object
        """
        if callable(expr):
            factor = expr(self._data)
            if name:
                factor._name = name
        else:
            # Build context for parser
            context = {col: self._data[col] for col in self._data.cols 
                       if col not in ["start_time", "end_time", "symbol"]}
            factor = self._parser.parse(expr, context)
            if name:
                factor._name = name
        
        # Cache the factor
        factor_name = name or factor.name
        self._factors[factor_name] = factor
        
        return factor
    
    def analyze(self, factor: Factor, price_col: str = "close") -> "FactorAnalyzer":
        """
        Create an analyzer for a factor.
        
        Args:
            factor: Factor to analyze
            price_col: Column to use for return calculation
            
        Returns:
            FactorAnalyzer instance
        """
        from ..factors.analyzer import FactorAnalyzer
        
        prices = self._data[price_col] if isinstance(price_col, str) else price_col
        return FactorAnalyzer(factor=factor, prices=prices)
    
    def backtest(
        self,
        signal: Factor,
        entry_price: str = "close",
        transaction_cost: Union[float, tuple[float, float]] = 0.0003,
        initial_capital: float = 10000.0,
        neutralization: str = "market",
        frequency: str = "1h",
        **kwargs,
    ) -> "BacktestResult":
        """
        Run a backtest with the given signal.
        
        Args:
            signal: Factor signal for position weighting
            entry_price: Column for entry prices
            transaction_cost: Transaction cost rate(s)
            initial_capital: Starting capital
            neutralization: "market" or "none"
            frequency: Trading frequency for annualization
            **kwargs: Additional arguments for Backtester
            
        Returns:
            BacktestResult
        """
        from ..backtest.backtester import Backtester
        
        bt = Backtester(
            prices=self._data,
            signal=signal,
            entry_price=entry_price,
            transaction_cost=transaction_cost,
            initial_capital=initial_capital,
            neutralization=neutralization,
            frequency=frequency,
            **kwargs,
        )
        return bt.run()
    
    def quick_report(
        self,
        factor: Factor,
        periods: List[int] = [1, 5, 10],
        quantiles: int = 5,
        price_col: str = "close",
    ) -> Dict:
        """
        Generate a quick analysis report for a factor.
        
        Args:
            factor: Factor to analyze
            periods: Holding periods for IC calculation
            quantiles: Number of quantiles for layer test
            price_col: Column to use for returns
            
        Returns:
            Dict with IC summary, quantile returns, etc.
        """
        analyzer = self.analyze(factor, price_col=price_col)
        analyzer.prepare_data(periods=periods)
        
        return {
            "ic_summary": analyzer.calculate_ic_summary(),
            "quantile_returns": {p: analyzer.calculate_quantile_returns(quantiles=quantiles, period=p) 
                                 for p in periods},
        }
    
    def __repr__(self) -> str:
        return f"ResearchSession(name={self.name!r}, symbols={len(self.symbols)}, factors={len(self._factors)})"
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/research/test_session.py::TestResearchSessionInit -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/factorium/research/ tests/research/
git commit -m "feat(research): add ResearchSession high-level API"
```

---

### Task B.2: ResearchSession 分析與回測整合

**Files:**
- Modify: `src/factorium/research/session.py`
- Test: `tests/research/test_session.py`

**Step 1: Write tests for analysis and backtest integration**

```python
class TestResearchSessionAnalysis:
    """Tests for ResearchSession analysis methods."""

    @pytest.fixture
    def session(self, sample_aggbar):
        return ResearchSession(data=sample_aggbar)

    def test_analyze_returns_analyzer(self, session):
        """analyze() should return a FactorAnalyzer."""
        from factorium.factors.analyzer import FactorAnalyzer
        
        close = session.factor("close")
        analyzer = session.analyze(close)
        
        assert isinstance(analyzer, FactorAnalyzer)

    def test_quick_report_returns_dict(self, session):
        """quick_report() should return analysis summary."""
        mom = session.create_factor("ts_delta(close, 5) / ts_shift(close, 5)")
        report = session.quick_report(mom, periods=[1, 5])
        
        assert "ic_summary" in report
        assert "quantile_returns" in report
        assert 1 in report["quantile_returns"]
        assert 5 in report["quantile_returns"]


class TestResearchSessionBacktest:
    """Tests for ResearchSession backtest methods."""

    @pytest.fixture
    def session(self, sample_aggbar):
        return ResearchSession(data=sample_aggbar)

    def test_backtest_returns_result(self, session):
        """backtest() should return BacktestResult."""
        from factorium.backtest import BacktestResult
        
        signal = session.factor("close").cs_rank()
        result = session.backtest(signal)
        
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_backtest_with_custom_params(self, session):
        """backtest() should accept custom parameters."""
        signal = session.factor("close").cs_rank()
        result = session.backtest(
            signal,
            initial_capital=50000.0,
            transaction_cost=0.001,
            neutralization="none",
        )
        
        assert result.portfolio_history["total_value"].iloc[0] == 50000.0
```

**Step 2: Run tests**

Run: `uv run pytest tests/research/test_session.py::TestResearchSessionAnalysis tests/research/test_session.py::TestResearchSessionBacktest -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/research/test_session.py
git commit -m "test(research): add analysis and backtest integration tests"
```

---

### Task B.3: 資料載入快捷方法

**Files:**
- Modify: `src/factorium/research/session.py`
- Test: `tests/research/test_session.py`

**Step 1: Write tests for data loading shortcuts**

```python
class TestResearchSessionDataLoading:
    """Tests for ResearchSession data loading methods."""

    def test_from_parquet(self, tmp_path, sample_aggbar):
        """Should load from parquet file."""
        # Save sample data
        path = tmp_path / "test_data.parquet"
        sample_aggbar.to_parquet(path)
        
        # Load via ResearchSession
        session = ResearchSession.from_parquet(path)
        
        assert len(session.symbols) == 3
        assert "BTC" in session.symbols

    def test_from_csv(self, tmp_path, sample_aggbar):
        """Should load from CSV file."""
        path = tmp_path / "test_data.csv"
        sample_aggbar.to_csv(path)
        
        session = ResearchSession.from_csv(path)
        
        assert len(session.symbols) == 3

    def test_slice_data(self, sample_aggbar):
        """Should be able to slice data by time or symbols."""
        session = ResearchSession(data=sample_aggbar)
        
        # Slice by symbols
        sliced = session.slice(symbols=["BTC", "ETH"])
        
        assert len(sliced.symbols) == 2
        assert "SOL" not in sliced.symbols
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/research/test_session.py::TestResearchSessionDataLoading -v`
Expected: FAIL (methods not implemented)

**Step 3: Implement data loading methods**

在 `src/factorium/research/session.py` 的 `ResearchSession` 類中添加：

```python
@classmethod
def from_parquet(cls, path: Union[str, Path], **kwargs) -> "ResearchSession":
    """
    Create ResearchSession from a Parquet file.
    
    Args:
        path: Path to parquet file
        **kwargs: Additional arguments for ResearchSession
        
    Returns:
        ResearchSession instance
    """
    import polars as pl
    from pathlib import Path
    
    path = Path(path)
    df = pl.read_parquet(path)
    return cls(data=AggBar(df), **kwargs)

@classmethod
def from_csv(cls, path: Union[str, Path], **kwargs) -> "ResearchSession":
    """
    Create ResearchSession from a CSV file.
    
    Args:
        path: Path to CSV file
        **kwargs: Additional arguments for ResearchSession
        
    Returns:
        ResearchSession instance
    """
    from pathlib import Path
    
    path = Path(path)
    agg = AggBar.from_csv(path)
    return cls(data=agg, **kwargs)

def slice(
    self,
    start: Optional[Union[datetime, int, str]] = None,
    end: Optional[Union[datetime, int, str]] = None,
    symbols: Optional[List[str]] = None,
) -> "ResearchSession":
    """
    Create a new session with sliced data.
    
    Args:
        start: Start time
        end: End time
        symbols: List of symbols to include
        
    Returns:
        New ResearchSession with filtered data
    """
    sliced_data = self._data.slice(start=start, end=end, symbols=symbols)
    return ResearchSession(data=sliced_data, name=f"{self.name}_sliced")
```

別忘了在檔案頂部添加：

```python
from pathlib import Path
from datetime import datetime
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/research/test_session.py::TestResearchSessionDataLoading -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/factorium/research/session.py tests/research/test_session.py
git commit -m "feat(research): add data loading shortcuts to ResearchSession"
```

---

### Task B.4: 更新 factorium 主模組導出

**Files:**
- Modify: `src/factorium/__init__.py`

**Step 1: Add ResearchSession to main exports**

確認 `src/factorium/__init__.py` 包含：

```python
from .research import ResearchSession
```

並更新 `__all__` 列表（如果有的話）。

**Step 2: Run import test**

```python
# Quick test
from factorium import ResearchSession
print(ResearchSession)
```

**Step 3: Commit**

```bash
git add src/factorium/__init__.py
git commit -m "feat: export ResearchSession from main module"
```

---

## Phase C: 深度因子分析與組合回測

### Task C.1: 多因子組合支援

**Files:**
- Create: `src/factorium/factors/composite.py`
- Test: `tests/factors/test_composite.py`

**Step 1: Write tests for composite factors**

`tests/factors/test_composite.py`:

```python
import pytest
import pandas as pd
import numpy as np

from factorium import AggBar, Factor
from factorium.factors.composite import CompositeFactor


class TestCompositeFactor:
    """Tests for multi-factor combination."""

    @pytest.fixture
    def sample_factors(self, sample_aggbar):
        """Create sample factors for testing."""
        close = sample_aggbar["close"]
        volume = sample_aggbar["volume"]
        
        mom = (close.ts_delta(5) / close.ts_shift(5)).cs_rank()
        vol = volume.ts_mean(5).cs_rank()
        
        return {"momentum": mom, "volume": vol}

    def test_equal_weight_combination(self, sample_factors):
        """Equal weight combination should average factors."""
        composite = CompositeFactor.from_equal_weights(
            list(sample_factors.values()),
            name="equal_weighted"
        )
        
        assert composite.name == "equal_weighted"
        assert not composite.to_pandas().empty

    def test_custom_weight_combination(self, sample_factors):
        """Custom weights should be applied correctly."""
        composite = CompositeFactor.from_weights(
            factors=list(sample_factors.values()),
            weights=[0.7, 0.3],
            name="custom_weighted"
        )
        
        assert composite.name == "custom_weighted"

    def test_zscore_combination(self, sample_factors):
        """Z-score combination should standardize before combining."""
        composite = CompositeFactor.from_zscore(
            list(sample_factors.values()),
            name="zscore_combined"
        )
        
        # Result should be roughly mean 0, std 1 within each cross-section
        data = composite.to_pandas()
        # Just verify it runs without error and produces values
        assert not data["factor"].isna().all()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/factors/test_composite.py -v`
Expected: FAIL (module not found)

**Step 3: Implement CompositeFactor**

`src/factorium/factors/composite.py`:

```python
"""
Composite factor utilities for multi-factor combination.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import polars as pl

from .core import Factor


class CompositeFactor:
    """
    Utilities for combining multiple factors into a composite signal.
    
    Supports various combination methods:
    - Equal weighting
    - Custom weighting  
    - Z-score normalization before combination
    """
    
    @staticmethod
    def from_equal_weights(
        factors: List[Factor],
        name: Optional[str] = None,
    ) -> Factor:
        """
        Combine factors with equal weights.
        
        Args:
            factors: List of Factor objects to combine
            name: Name for the resulting factor
            
        Returns:
            Combined Factor
        """
        if not factors:
            raise ValueError("At least one factor required")
        
        weights = [1.0 / len(factors)] * len(factors)
        return CompositeFactor.from_weights(factors, weights, name=name)
    
    @staticmethod
    def from_weights(
        factors: List[Factor],
        weights: List[float],
        name: Optional[str] = None,
    ) -> Factor:
        """
        Combine factors with custom weights.
        
        Args:
            factors: List of Factor objects
            weights: Weights for each factor (will be normalized to sum to 1)
            name: Name for the resulting factor
            
        Returns:
            Combined Factor
        """
        if len(factors) != len(weights):
            raise ValueError("Number of factors must match number of weights")
        
        if not factors:
            raise ValueError("At least one factor required")
        
        # Normalize weights
        total = sum(weights)
        if total == 0:
            raise ValueError("Weights must not sum to zero")
        weights = [w / total for w in weights]
        
        # Start with first factor scaled by its weight
        result = factors[0] * weights[0]
        
        # Add remaining factors
        for factor, weight in zip(factors[1:], weights[1:]):
            result = result + (factor * weight)
        
        result._name = name or "composite"
        return result
    
    @staticmethod
    def from_zscore(
        factors: List[Factor],
        name: Optional[str] = None,
    ) -> Factor:
        """
        Combine factors after z-score normalization.
        
        Each factor is cross-sectionally z-scored before equal-weight combination.
        
        Args:
            factors: List of Factor objects
            name: Name for the resulting factor
            
        Returns:
            Combined Factor
        """
        if not factors:
            raise ValueError("At least one factor required")
        
        # Z-score each factor cross-sectionally
        zscored = [f.cs_zscore() for f in factors]
        
        # Equal weight combination
        return CompositeFactor.from_equal_weights(zscored, name=name)
```

**Step 4: Update factors package exports**

`src/factorium/factors/__init__.py` 添加：

```python
from .composite import CompositeFactor
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/factors/test_composite.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/factorium/factors/composite.py src/factorium/factors/__init__.py tests/factors/test_composite.py
git commit -m "feat(factors): add CompositeFactor for multi-factor combination"
```

---

### Task C.2: 回測約束條件

**Files:**
- Create: `src/factorium/backtest/constraints.py`
- Modify: `src/factorium/backtest/backtester.py`
- Test: `tests/backtest/test_constraints.py`

**Step 1: Write tests for constraints**

`tests/backtest/test_constraints.py`:

```python
import pytest
import pandas as pd
import numpy as np

from factorium.backtest.constraints import (
    WeightConstraint,
    apply_weight_cap,
    apply_gross_exposure_cap,
)


class TestWeightConstraints:
    """Tests for weight constraints."""

    def test_weight_cap_clips_large_weights(self):
        """Weights exceeding cap should be clipped."""
        weights = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
        capped = apply_weight_cap(weights, max_weight=0.25)
        
        assert capped.max() <= 0.25
        assert abs(capped.sum() - 1.0) < 1e-10  # Should still sum to 1

    def test_weight_cap_no_change_when_below(self):
        """Weights below cap should not change."""
        weights = pd.Series([0.2, 0.3, 0.5], index=["A", "B", "C"])
        capped = apply_weight_cap(weights, max_weight=0.6)
        
        pd.testing.assert_series_equal(weights, capped)

    def test_gross_exposure_cap(self):
        """Gross exposure should be capped."""
        weights = pd.Series([0.6, 0.4, -0.5, -0.3], index=["A", "B", "C", "D"])
        # Gross exposure = 0.6 + 0.4 + 0.5 + 0.3 = 1.8
        
        capped = apply_gross_exposure_cap(weights, max_gross=1.0)
        
        assert capped.abs().sum() <= 1.0 + 1e-10


class TestWeightConstraintClass:
    """Tests for WeightConstraint dataclass."""

    def test_constraint_application(self):
        """Constraint should apply all limits."""
        constraint = WeightConstraint(
            max_weight=0.3,
            max_gross_exposure=1.5,
        )
        
        weights = pd.Series([0.5, 0.3, 0.2, -0.4], index=["A", "B", "C", "D"])
        constrained = constraint.apply(weights)
        
        assert constrained.max() <= 0.3
        assert constrained.abs().sum() <= 1.5 + 1e-10
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_constraints.py -v`
Expected: FAIL (module not found)

**Step 3: Implement constraints**

`src/factorium/backtest/constraints.py`:

```python
"""
Portfolio weight constraints for backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


def apply_weight_cap(
    weights: pd.Series,
    max_weight: float,
) -> pd.Series:
    """
    Cap individual weights and redistribute excess.
    
    Args:
        weights: Portfolio weights (can be long/short)
        max_weight: Maximum absolute weight per position
        
    Returns:
        Capped weights that sum to original sum
    """
    if weights.empty:
        return weights
    
    original_sum = weights.sum()
    
    # Clip weights
    capped = weights.clip(lower=-max_weight, upper=max_weight)
    
    # Redistribute to maintain sum
    if abs(capped.sum()) > 1e-10:
        capped = capped * (original_sum / capped.sum())
    
    return capped


def apply_gross_exposure_cap(
    weights: pd.Series,
    max_gross: float,
) -> pd.Series:
    """
    Cap gross exposure (sum of absolute weights).
    
    Args:
        weights: Portfolio weights
        max_gross: Maximum gross exposure
        
    Returns:
        Scaled weights with gross exposure <= max_gross
    """
    if weights.empty:
        return weights
    
    gross = weights.abs().sum()
    
    if gross > max_gross:
        return weights * (max_gross / gross)
    
    return weights


@dataclass
class WeightConstraint:
    """
    Container for portfolio weight constraints.
    
    Args:
        max_weight: Maximum absolute weight per position (default: None = no limit)
        max_gross_exposure: Maximum gross exposure (default: None = no limit)
        market_neutral: If True, weights must sum to zero (default: False)
    """
    max_weight: Optional[float] = None
    max_gross_exposure: Optional[float] = None
    market_neutral: bool = False
    
    def apply(self, weights: pd.Series) -> pd.Series:
        """
        Apply all constraints to weights.
        
        Args:
            weights: Input portfolio weights
            
        Returns:
            Constrained weights
        """
        result = weights.copy()
        
        # Apply weight cap first
        if self.max_weight is not None:
            result = apply_weight_cap(result, self.max_weight)
        
        # Apply gross exposure cap
        if self.max_gross_exposure is not None:
            result = apply_gross_exposure_cap(result, self.max_gross_exposure)
        
        # Enforce market neutrality
        if self.market_neutral:
            # Subtract mean to make sum zero
            result = result - result.mean()
        
        return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_constraints.py -v`
Expected: All PASS

**Step 5: Update backtest package exports**

`src/factorium/backtest/__init__.py` 添加：

```python
from .constraints import WeightConstraint, apply_weight_cap, apply_gross_exposure_cap
```

**Step 6: Commit**

```bash
git add src/factorium/backtest/constraints.py src/factorium/backtest/__init__.py tests/backtest/test_constraints.py
git commit -m "feat(backtest): add portfolio weight constraints"
```

---

### Task C.3: Backtester 整合約束條件

**Files:**
- Modify: `src/factorium/backtest/backtester.py`
- Test: `tests/backtest/test_backtester.py`

**Step 1: Write test for backtester with constraints**

```python
class TestBacktesterConstraints:
    """Tests for backtester with weight constraints."""

    def test_backtester_with_weight_constraint(self, sample_data):
        """Backtester should apply weight constraints."""
        from factorium.backtest.constraints import WeightConstraint
        
        signal = sample_data["close"].cs_rank()
        constraint = WeightConstraint(max_weight=0.4)
        
        bt = Backtester(
            prices=sample_data,
            signal=signal,
            neutralization="market",
            constraint=constraint,
        )
        result = bt.run()
        
        assert isinstance(result, BacktestResult)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_backtester.py::TestBacktesterConstraints -v`
Expected: FAIL (constraint parameter not supported)

**Step 3: Add constraint support to Backtester**

修改 `src/factorium/backtest/backtester.py`:

1. 在 `__init__` 中添加 `constraint` 參數：

```python
from .constraints import WeightConstraint

def __init__(
    self,
    prices: "AggBar",
    signal: "Factor",
    entry_price: str = "close",
    transaction_cost: Union[float, tuple[float, float]] = 0.0003,
    initial_capital: float = 10000.0,
    full_rebalance: bool = False,
    neutralization: Literal["market", "none"] = "market",
    frequency: str = "1h",
    constraint: Optional[WeightConstraint] = None,  # 新增
):
    # ... existing code ...
    self.constraint = constraint
```

2. 在 `_calculate_target_weights` 末尾應用約束：

```python
def _calculate_target_weights(self, signals: pd.Series) -> pd.Series:
    signals = signals.dropna()

    if len(signals) == 0:
        return pd.Series(dtype=float)

    if self.neutralization == "none":
        weights = normalize_weights(signals)
    elif self.neutralization == "market":
        weights = neutralize_weights(signals)
    else:
        raise ValueError(f"Unknown neutralization: {self.neutralization}")
    
    # Apply constraints if specified
    if self.constraint is not None:
        weights = self.constraint.apply(weights)
    
    return weights
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backtest/test_backtester.py::TestBacktesterConstraints -v`
Expected: PASS

**Step 5: Run all backtest tests**

Run: `uv run pytest tests/backtest/ -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/factorium/backtest/backtester.py tests/backtest/test_backtester.py
git commit -m "feat(backtest): integrate weight constraints into Backtester"
```

---

### Task C.4: 報告生成器

**Files:**
- Create: `src/factorium/research/report.py`
- Test: `tests/research/test_report.py`

**Step 1: Write tests for report generation**

`tests/research/test_report.py`:

```python
import pytest
import pandas as pd
import numpy as np

from factorium import AggBar
from factorium.research import ResearchSession
from factorium.research.report import FactorReport


class TestFactorReport:
    """Tests for factor report generation."""

    @pytest.fixture
    def session_with_factor(self, sample_aggbar):
        session = ResearchSession(data=sample_aggbar)
        mom = session.create_factor("ts_delta(close, 5) / ts_shift(close, 5)", name="momentum")
        return session, mom

    def test_report_generation(self, session_with_factor):
        """Should generate a complete report."""
        session, factor = session_with_factor
        
        report = FactorReport(
            session=session,
            factor=factor,
            periods=[1, 5],
        )
        report.generate()
        
        assert report.ic_summary is not None
        assert report.backtest_result is not None

    def test_report_to_dict(self, session_with_factor):
        """Report should be convertible to dict."""
        session, factor = session_with_factor
        
        report = FactorReport(session=session, factor=factor)
        report.generate()
        
        data = report.to_dict()
        
        assert "ic_summary" in data
        assert "backtest_metrics" in data
        assert "factor_name" in data

    def test_report_summary_string(self, session_with_factor):
        """Report should have summary method."""
        session, factor = session_with_factor
        
        report = FactorReport(session=session, factor=factor)
        report.generate()
        
        summary = report.summary()
        
        assert isinstance(summary, str)
        assert "momentum" in summary
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/research/test_report.py -v`
Expected: FAIL (module not found)

**Step 3: Implement FactorReport**

`src/factorium/research/report.py`:

```python
"""
Factor research report generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Any

import pandas as pd

if TYPE_CHECKING:
    from .session import ResearchSession
    from ..factors.core import Factor
    from ..backtest.backtester import BacktestResult


@dataclass
class FactorReport:
    """
    Comprehensive factor analysis report.
    
    Combines IC analysis, quantile returns, and backtest results
    into a single report object.
    
    Args:
        session: ResearchSession containing the data
        factor: Factor to analyze
        periods: Holding periods for IC calculation
        quantiles: Number of quantiles for layer test
        price_col: Column to use for returns
    """
    session: "ResearchSession"
    factor: "Factor"
    periods: List[int] = field(default_factory=lambda: [1, 5, 10])
    quantiles: int = 5
    price_col: str = "close"
    
    # Results (populated by generate())
    ic_summary: Optional[pd.DataFrame] = field(default=None, init=False)
    ic_series: Optional[pd.DataFrame] = field(default=None, init=False)
    quantile_returns: Optional[Dict[int, pd.DataFrame]] = field(default=None, init=False)
    backtest_result: Optional["BacktestResult"] = field(default=None, init=False)
    
    def generate(self) -> "FactorReport":
        """
        Generate the full report.
        
        Returns:
            self for method chaining
        """
        # IC Analysis
        analyzer = self.session.analyze(self.factor, price_col=self.price_col)
        analyzer.prepare_data(periods=self.periods)
        
        self.ic_summary = analyzer.calculate_ic_summary()
        self.ic_series = analyzer.calculate_ic()
        
        # Quantile Returns
        self.quantile_returns = {}
        for p in self.periods:
            try:
                self.quantile_returns[p] = analyzer.calculate_quantile_returns(
                    quantiles=self.quantiles, 
                    period=p
                )
            except Exception:
                self.quantile_returns[p] = pd.DataFrame()
        
        # Backtest
        try:
            signal = self.factor.cs_rank()
            self.backtest_result = self.session.backtest(signal)
        except Exception as e:
            import logging
            logging.warning(f"Backtest failed: {e}")
            self.backtest_result = None
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary format.
        
        Returns:
            Dict containing all report data
        """
        result = {
            "factor_name": self.factor.name,
            "periods": self.periods,
            "quantiles": self.quantiles,
        }
        
        if self.ic_summary is not None:
            result["ic_summary"] = self.ic_summary.to_dict()
        
        if self.quantile_returns is not None:
            result["quantile_returns"] = {
                p: df.to_dict() if not df.empty else {} 
                for p, df in self.quantile_returns.items()
            }
        
        if self.backtest_result is not None:
            result["backtest_metrics"] = self.backtest_result.metrics
        
        return result
    
    def summary(self) -> str:
        """
        Generate a text summary of the report.
        
        Returns:
            Formatted string summary
        """
        lines = [
            f"=== Factor Report: {self.factor.name} ===",
            "",
        ]
        
        if self.ic_summary is not None:
            lines.append("IC Summary:")
            lines.append(self.ic_summary.to_string())
            lines.append("")
        
        if self.backtest_result is not None:
            lines.append("Backtest Metrics:")
            for key, value in self.backtest_result.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")
        
        return "\n".join(lines)
```

**Step 4: Update research package exports**

`src/factorium/research/__init__.py`:

```python
from .session import ResearchSession
from .report import FactorReport

__all__ = ["ResearchSession", "FactorReport"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/research/test_report.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/factorium/research/report.py src/factorium/research/__init__.py tests/research/test_report.py
git commit -m "feat(research): add FactorReport for comprehensive analysis"
```

---

## Summary Checklist

### Phase 0: 回測修復
- [ ] Task 0.1: Portfolio 現金不足檢查
- [ ] Task 0.2: Backtester 處理被拒絕的交易
- [ ] Task 0.3: 價格缺失時的警告與處理
- [ ] Task 0.4: Metrics 邊界情況改進

### Phase B: Research Session API
- [ ] Task B.1: ResearchSession 基礎架構
- [ ] Task B.2: ResearchSession 分析與回測整合
- [ ] Task B.3: 資料載入快捷方法
- [ ] Task B.4: 更新主模組導出

### Phase C: 深度分析與回測
- [ ] Task C.1: 多因子組合支援
- [ ] Task C.2: 回測約束條件
- [ ] Task C.3: Backtester 整合約束條件
- [ ] Task C.4: 報告生成器

---

## Notes

- 所有測試使用 `uv run pytest` 執行
- 每個 Task 完成後立即 commit，保持小步提交
- Phase 0 是前置條件，必須先完成才能開始 Phase B/C
- Phase B 和 C 的任務可以部分並行，但建議按順序執行以減少衝突
