# Return-Based VectorizedBacktester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert VectorizedBacktester from qty-based to return-based backtest, fixing critical review findings (#1-#3, #8, #12).

**Architecture:** Replace position-tracking (target_qty, trade_qty, cash) with weight-based portfolio returns: `portfolio_return = sum(weight * asset_return) - cost`. Equity is accumulated multiplicatively. Constraints are followed by a renormalization step.

**Tech Stack:** Python 3.13, Polars, Pandas (metrics only), pytest

**Spec:** `docs/superpowers/specs/2026-04-05-return-based-backtest-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/factorium/backtest/utils.py` | Modify | Add `renormalize_weights()` |
| `src/factorium/backtest/constraints.py` | Modify | Fix `MaxGrossExposureConstraint` duplicate join |
| `src/factorium/backtest/vectorized.py` | Rewrite | Return-based core: `BacktestResult`, `BacktestResultPandas`, all `_calculate_*` methods, `summary()`, `to_pandas()` |
| `src/factorium/backtest/__init__.py` | Modify | Add `BacktestResultPandas` to `__all__` |
| `tests/backtest/test_backtester.py` | Modify | Update VectorizedBacktester tests for new output |
| `tests/universe/test_backtest_mask.py` | Modify | Minor: adapt to new result structure |
| `docs/user-guide/backtest.md` | Modify | Update output docs |

---

### Task 1: Add `renormalize_weights()` to utils.py

**Files:**
- Modify: `src/factorium/backtest/utils.py`
- Test: `tests/backtest/test_backtester.py` (add new test class)

- [ ] **Step 1: Write failing tests for renormalize_weights**

Add at the top of `tests/backtest/test_backtester.py` imports and a new test class after `TestNormalizeWeights`:

```python
# Add to imports at top of file:
from factorium.backtest.utils import renormalize_weights

# Add new test class after TestNormalizeWeights (around line 61):
class TestRenormalizeWeights:
    """Tests for post-constraint weight renormalization."""

    def test_market_neutral_sum_zero_abs_one(self):
        """After renormalization, market neutral weights sum to 0 with abs sum 1."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "weight": [0.3, 0.1, -0.05, -0.1],  # sum != 0, abs_sum != 1
        })
        result = renormalize_weights(df, neutralization="market")
        weights = result["weight"]
        assert abs(weights.sum()) < 1e-10
        assert abs(weights.abs().sum() - 1.0) < 1e-10

    def test_long_only_sum_one_all_positive(self):
        """After renormalization, long-only weights sum to 1 and are all >= 0."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.5, 0.3, -0.1],  # has a negative weight
        })
        result = renormalize_weights(df, neutralization="none")
        weights = result["weight"]
        assert abs(weights.sum() - 1.0) < 1e-10
        assert (weights >= -1e-10).all()

    def test_all_zero_weights_stay_zero(self):
        """If all weights are zero, they should stay zero."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.0, 0.0, 0.0],
        })
        result_market = renormalize_weights(df, neutralization="market")
        assert result_market["weight"].abs().sum() == 0.0
        result_long = renormalize_weights(df, neutralization="none")
        assert result_long["weight"].abs().sum() == 0.0

    def test_multiple_timestamps(self):
        """Renormalization should work per-timestamp independently."""
        df = pl.DataFrame({
            "end_time": [1000, 1000, 2000, 2000],
            "symbol": ["A", "B", "A", "B"],
            "weight": [0.6, -0.2, 0.3, 0.1],
        })
        result = renormalize_weights(df, neutralization="market")
        for t in [1000, 2000]:
            subset = result.filter(pl.col("end_time") == t)["weight"]
            assert abs(subset.sum()) < 1e-10
            assert abs(subset.abs().sum() - 1.0) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/raid1/novis/factorium && uv run pytest tests/backtest/test_backtester.py::TestRenormalizeWeights -v`
Expected: FAIL with `ImportError: cannot import name 'renormalize_weights'`

- [ ] **Step 3: Implement renormalize_weights in utils.py**

Add at the end of `src/factorium/backtest/utils.py` (after `safe_divide`):

```python
def renormalize_weights(df: pl.DataFrame, neutralization: str) -> pl.DataFrame:
    """
    Renormalize weights after constraint application.

    For market neutral: demean then scale so sum(w)=0, sum(|w|)=1 per timestamp.
    For long-only (none): clip negatives to 0, scale so sum(w)=1 per timestamp.

    Args:
        df: DataFrame with columns [end_time, symbol, weight]
        neutralization: "market" or "none"

    Returns:
        DataFrame with renormalized weight column
    """
    if neutralization == "market":
        # Demean per timestamp
        df = df.with_columns(
            (pl.col("weight") - pl.col("weight").mean().over("end_time")).alias("weight")
        )
        # Scale by abs sum per timestamp
        abs_sum = pl.col("weight").abs().sum().over("end_time")
        df = df.with_columns(
            pl.when(abs_sum > EPSILON)
            .then(pl.col("weight") / abs_sum)
            .otherwise(0.0)
            .alias("weight")
        )
    else:
        # Clip negatives
        df = df.with_columns(
            pl.when(pl.col("weight") < 0.0)
            .then(0.0)
            .otherwise(pl.col("weight"))
            .alias("weight")
        )
        # Scale to sum=1 per timestamp
        w_sum = pl.col("weight").sum().over("end_time")
        df = df.with_columns(
            pl.when(w_sum > EPSILON)
            .then(pl.col("weight") / w_sum)
            .otherwise(0.0)
            .alias("weight")
        )
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/raid1/novis/factorium && uv run pytest tests/backtest/test_backtester.py::TestRenormalizeWeights -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/factorium/backtest/utils.py tests/backtest/test_backtester.py
git commit -m "feat(backtest): add renormalize_weights for post-constraint normalization"
```

---

### Task 2: Fix MaxGrossExposureConstraint duplicate join

**Files:**
- Modify: `src/factorium/backtest/constraints.py:85-98`
- Test: `tests/backtest/test_constraints.py`

- [ ] **Step 1: Write failing test for duplicate apply**

Add to `tests/backtest/test_constraints.py` inside `TestMaxGrossExposureConstraint`:

```python
    def test_double_apply_no_duplicate_columns(self):
        """Applying the constraint twice should not create duplicate columns."""
        weights = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.6, 0.5, -0.3],  # gross = 1.4
        })
        constraint = MaxGrossExposureConstraint(max_exposure=1.0)
        result = constraint.apply(weights)
        result2 = constraint.apply(result)
        assert "gross" not in result2.columns
        assert "gross_right" not in result2.columns
        gross = result2["weight"].abs().sum()
        assert abs(gross - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/raid1/novis/factorium && uv run pytest tests/backtest/test_constraints.py::TestMaxGrossExposureConstraint::test_double_apply_no_duplicate_columns -v`
Expected: FAIL — `gross_right` column exists or wrong result

- [ ] **Step 3: Fix MaxGrossExposureConstraint.apply**

In `src/factorium/backtest/constraints.py`, replace the `apply` method of `MaxGrossExposureConstraint` (lines 85-98):

```python
    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        # Drop gross column if it already exists (idempotent)
        if "gross" in weights.columns:
            weights = weights.drop("gross")

        # Group by end_time, calculate gross, scale if needed
        gross = weights.group_by("end_time").agg(pl.col("weight").abs().sum().alias("gross"))

        weights = weights.join(gross, on="end_time")

        weights = weights.with_columns(
            pl.when(pl.col("gross") > self.max_exposure)
            .then(pl.col("weight") * self.max_exposure / pl.col("gross"))
            .otherwise(pl.col("weight"))
            .alias("weight")
        )

        return weights.drop("gross")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/raid1/novis/factorium && uv run pytest tests/backtest/test_constraints.py -v`
Expected: All passed (including new test)

- [ ] **Step 5: Commit**

```bash
git add src/factorium/backtest/constraints.py tests/backtest/test_constraints.py
git commit -m "fix(backtest): prevent duplicate column in MaxGrossExposureConstraint"
```

---

### Task 3: Rewrite BacktestResult and BacktestResultPandas dataclasses

**Files:**
- Modify: `src/factorium/backtest/vectorized.py:17-47`
- Modify: `src/factorium/backtest/__init__.py`

- [ ] **Step 1: Write failing test for new output structure**

Add to `tests/backtest/test_backtester.py`, replacing the `test_vectorized_polars_output_types` test in `TestVectorizedBacktesterIntegration`:

```python
    def test_vectorized_polars_output_types(self, sample_data):
        """VectorizedBacktester should return Polars DataFrames with correct fields."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        assert isinstance(result.equity_curve, pl.DataFrame)
        assert isinstance(result.returns, pl.DataFrame)
        assert isinstance(result.weights, pl.DataFrame)
        assert isinstance(result.turnover, pl.DataFrame)

        # Check column names
        assert set(result.equity_curve.columns) == {"end_time", "total_value"}
        assert set(result.returns.columns) == {"end_time", "return"}
        assert set(result.weights.columns) == {"end_time", "symbol", "weight"}
        assert set(result.turnover.columns) == {"end_time", "turnover", "cost"}

        # Should NOT have trades or portfolio_history
        assert not hasattr(result, "trades")
        assert not hasattr(result, "portfolio_history")
```

Also add a test for `BacktestResultPandas` import:

```python
    def test_backtest_result_pandas_importable(self):
        """BacktestResultPandas should be importable from factorium.backtest."""
        from factorium.backtest import BacktestResultPandas
        assert BacktestResultPandas is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/raid1/novis/factorium && uv run pytest tests/backtest/test_backtester.py::TestVectorizedBacktesterIntegration::test_vectorized_polars_output_types tests/backtest/test_backtester.py::TestVectorizedBacktesterIntegration::test_backtest_result_pandas_importable -v`
Expected: FAIL — `result.weights` does not exist, `BacktestResultPandas` not in `__all__`

- [ ] **Step 3: Rewrite dataclasses in vectorized.py**

Replace lines 17-47 in `src/factorium/backtest/vectorized.py`:

```python
@dataclass
class BacktestResult:
    """Results from a backtest run."""

    equity_curve: pl.DataFrame  # columns: [end_time, total_value]
    returns: pl.DataFrame  # columns: [end_time, return]
    weights: pl.DataFrame  # columns: [end_time, symbol, weight]
    turnover: pl.DataFrame  # columns: [end_time, turnover, cost]
    metrics: dict[str, float]

    def to_pandas(self) -> "BacktestResultPandas":
        """Convert all DataFrames to pandas for backward compatibility."""
        return BacktestResultPandas(
            equity_curve=self.equity_curve.to_pandas(),
            returns=self.returns.to_pandas(),
            weights=self.weights.to_pandas(),
            turnover=self.turnover.to_pandas(),
            metrics=self.metrics,
        )


@dataclass
class BacktestResultPandas:
    """Pandas version of BacktestResult for backward compatibility."""

    equity_curve: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.DataFrame
    metrics: dict[str, float]
```

- [ ] **Step 4: Add BacktestResultPandas to __init__.py __all__**

In `src/factorium/backtest/__init__.py`, add the import and `__all__` entry:

```python
from .vectorized import BacktestResult, BacktestResultPandas, VectorizedBacktester
```

Add `"BacktestResultPandas"` to the `__all__` list (after `"BacktestResult"`).

- [ ] **Step 5: Do NOT run tests yet** — the rest of `vectorized.py` still references `trades`/`portfolio_history`. This task just updates the dataclasses. We'll fix the rest in Task 4. Commit the dataclass + __init__ changes:

```bash
git add src/factorium/backtest/vectorized.py src/factorium/backtest/__init__.py
git commit -m "refactor(backtest): update BacktestResult to return-based output structure"
```

---

### Task 4: Rewrite VectorizedBacktester core methods

This is the main task. Replace `_calculate_positions`, `_calculate_equity`, `_calculate_metrics`, `_build_result`, and `summary()`.

**Files:**
- Modify: `src/factorium/backtest/vectorized.py`

- [ ] **Step 1: Rewrite `_prepare_data` to include asset returns**

In `src/factorium/backtest/vectorized.py`, replace the `_prepare_data` method (lines 151-171). The new version adds asset return calculation:

```python
    def _prepare_data(self) -> pl.DataFrame:
        """Merge prices and signals, shift signals to avoid lookahead bias."""
        # Get the entry price column
        price_cols = ["end_time", "symbol", self.entry_price]
        if self._mask is not None:
            price_cols.append(self._mask)
        prices_df = self.prices_df.select(price_cols).rename({self.entry_price: "price"})

        # Prepare signal data
        signal_df = self.signal_df.select(["end_time", "symbol", "factor"]).rename({"factor": "signal"})

        # Join on end_time and symbol
        combined = prices_df.join(signal_df, on=["end_time", "symbol"], how="left")

        # Shift signal by 1 per symbol to use previous signal (avoid lookahead bias)
        combined = combined.with_columns([
            pl.col("signal").shift(1).over("symbol").alias("prev_signal")
        ]).drop("signal")

        # Calculate asset returns: r_t = price_t / price_{t-1} - 1
        combined = combined.with_columns([
            (pl.col("price") / pl.col("price").shift(1).over("symbol") - 1.0).alias("asset_return")
        ])

        # Sort for stable processing
        combined = combined.sort(["end_time", "symbol"])

        return combined
```

- [ ] **Step 2: Add renormalization call to `_calculate_weights`**

In `_calculate_weights`, add renormalization after constraints. Replace lines 200-204:

```python
        # Apply constraints
        for constraint in self.constraints:
            df = constraint.apply(df)

        # Renormalize weights after constraints
        from .utils import renormalize_weights
        df = renormalize_weights(df, neutralization=self.neutralization)

        return df
```

- [ ] **Step 3: Replace `_calculate_positions` with `_calculate_returns`**

Delete the entire `_calculate_positions` method (lines 206-228) and replace with:

```python
    def _calculate_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate portfolio returns from weights and asset returns."""
        # Previous weight per symbol (for turnover calculation)
        df = df.with_columns([
            pl.col("weight").shift(1).over("symbol").fill_null(0.0).alias("prev_weight")
        ])

        # Weight change per symbol
        df = df.with_columns([
            (pl.col("weight") - pl.col("prev_weight")).alias("weight_change")
        ])

        # Per-symbol contribution to portfolio return
        df = df.with_columns([
            (pl.col("weight") * pl.col("asset_return").fill_null(0.0)).alias("contribution")
        ])

        return df
```

- [ ] **Step 4: Rewrite `_calculate_equity`**

Delete the entire `_calculate_equity` method (lines 230-272) and replace with:

```python
    def _calculate_equity(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Calculate portfolio equity and turnover from per-symbol data.

        Returns:
            Tuple of (equity_df, turnover_df).
            equity_df columns: [end_time, total_value]
            turnover_df columns: [end_time, turnover, cost]
        """
        buy_rate, sell_rate = self.transaction_cost

        # Aggregate to time level
        per_period = (
            df.group_by("end_time")
            .agg([
                pl.col("contribution").sum().alias("gross_return"),
                # Buy-side turnover: sum of positive weight changes
                pl.col("weight_change").clip(lower_bound=0.0).sum().alias("buy_turnover"),
                # Sell-side turnover: sum of abs(negative weight changes)
                (-pl.col("weight_change").clip(upper_bound=0.0)).sum().alias("sell_turnover"),
            ])
            .sort("end_time")
        )

        # Calculate cost and net return
        per_period = per_period.with_columns([
            (pl.col("buy_turnover") + pl.col("sell_turnover")).alias("turnover"),
            (pl.col("buy_turnover") * buy_rate + pl.col("sell_turnover") * sell_rate).alias("cost"),
        ]).with_columns([
            (pl.col("gross_return") - pl.col("cost")).alias("net_return"),
        ])

        # Cumulative equity: equity_t = initial_capital * prod(1 + net_return)
        per_period = per_period.with_columns([
            (
                (1.0 + pl.col("net_return")).cum_prod() * self.initial_capital
            ).alias("total_value")
        ])

        equity_df = per_period.select(["end_time", "total_value"])
        turnover_df = per_period.select(["end_time", "turnover", "cost"])

        return equity_df, turnover_df
```

- [ ] **Step 5: Simplify `_calculate_metrics`**

Delete the entire `_calculate_metrics` method (lines 274-324) and replace with:

```python
    def _calculate_metrics(self, equity_df: pl.DataFrame) -> dict[str, float]:
        """Calculate performance metrics by delegating to calculate_metrics()."""
        equity_pd = equity_df.to_pandas()
        returns_series = equity_pd["total_value"].pct_change().dropna()

        if len(returns_series) < 2:
            return {
                "total_return": np.nan,
                "annual_return": np.nan,
                "annual_volatility": np.nan,
                "sharpe_ratio": np.nan,
                "sortino_ratio": np.nan,
                "calmar_ratio": np.nan,
                "max_drawdown": np.nan,
                "var_95": np.nan,
                "cvar_95": np.nan,
                "win_rate": np.nan,
                "profit_factor": np.nan,
            }

        return calculate_metrics(
            returns_series,
            risk_free_rate=0.0,
            periods_per_year=self.periods_per_year,
        )
```

- [ ] **Step 6: Rewrite `_build_result`**

Delete the entire `_build_result` method (lines 326-358) and replace with:

```python
    def _build_result(
        self,
        equity_df: pl.DataFrame,
        turnover_df: pl.DataFrame,
        weights_df: pl.DataFrame,
    ) -> BacktestResult:
        """Assemble final result."""
        # Returns
        returns = (
            equity_df
            .with_columns([pl.col("total_value").pct_change().alias("return")])
            .select(["end_time", "return"])
        )

        # Calculate metrics
        metrics = self._calculate_metrics(equity_df)

        return BacktestResult(
            equity_curve=equity_df,
            returns=returns,
            weights=weights_df,
            turnover=turnover_df,
            metrics=metrics,
        )
```

- [ ] **Step 7: Update `run()` to use new methods**

Replace the `run` method body (lines 114-135):

```python
    def run(self) -> BacktestResult:
        """
        Run the backtest and return results.

        Returns:
            BacktestResult with equity_curve, returns, weights, turnover, and metrics
        """
        # Step 1: Prepare data (prices, signals, asset returns)
        combined = self._prepare_data()

        # Step 2: Calculate weights (neutralization + constraints + renormalization)
        combined = self._calculate_weights(combined)

        # Step 3: Calculate per-symbol returns
        combined = self._calculate_returns(combined)

        # Step 4: Calculate equity and turnover
        equity_df, turnover_df = self._calculate_equity(combined)

        # Step 5: Extract final weights
        weights_df = combined.select(["end_time", "symbol", "weight"])

        # Step 6: Build result
        self._result = self._build_result(equity_df, turnover_df, weights_df)
        return self._result
```

- [ ] **Step 8: Update `summary()`**

Replace the `summary` method (lines 137-149):

```python
    def summary(self) -> dict[str, Any]:
        """Return a summary of backtest results."""
        if self._result is None:
            raise RuntimeError("Must call run() before summary()")

        final_value = self._result.equity_curve["total_value"].to_list()[-1]

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_turnover": float(self._result.turnover["turnover"].sum()),
            "total_cost": float(self._result.turnover["cost"].sum()),
            **self._result.metrics,
        }
```

- [ ] **Step 9: Remove unused imports**

In `vectorized.py`, the import of `EPSILON` from `..constants` is no longer needed in this file (it's used in utils.py). Check and remove if unused. `numpy` may also be unused now — check and remove. Keep `np` only if still referenced.

After this step, `vectorized.py` imports should be:

```python
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from ..aggbar import AggBar
from ..factors.core import Factor
from .metrics import calculate_metrics
from .utils import frequency_to_periods_per_year
```

(`numpy` is still needed for the `np.nan` fallback in `_calculate_metrics`. `EPSILON` import is removed — it's only used in `utils.py` now.)

- [ ] **Step 10: Run the full test suite to see what still fails**

Run: `cd /mnt/raid1/novis/factorium && uv run pytest tests/backtest/ tests/universe/test_backtest_mask.py -v 2>&1 | head -80`
Expected: Some tests fail (we'll fix those in Task 5). The core engine should work.

- [ ] **Step 11: Commit**

```bash
git add src/factorium/backtest/vectorized.py
git commit -m "feat(backtest): rewrite VectorizedBacktester to return-based simulation"
```

---

### Task 5: Update existing tests for new output structure

**Files:**
- Modify: `tests/backtest/test_backtester.py`
- Modify: `tests/universe/test_backtest_mask.py`

- [ ] **Step 1: Update TestBacktester.test_summary**

In `tests/backtest/test_backtester.py`, replace `test_summary` (lines 185-197):

```python
    def test_summary(self, sample_data):
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = Backtester(prices=sample_data, signal=signal)
        bt.run()

        summary = bt.summary()

        assert "initial_capital" in summary
        assert "final_value" in summary
        assert "total_turnover" in summary
        assert "total_cost" in summary
        assert "sharpe_ratio" in summary
        # num_trades is removed in return-based mode
        assert "num_trades" not in summary
```

- [ ] **Step 2: Update test_no_lookahead_bias to use weights**

Replace `test_no_lookahead_bias` (lines 199-210):

```python
    def test_no_lookahead_bias(self, sample_data):
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = Backtester(prices=sample_data, signal=signal)
        result = bt.run()

        # Weights should exist and have data
        assert len(result.weights) > 0
        # First timestamp should have zero weights (no prev signal)
        first_time = result.weights["end_time"].min()
        first_weights = result.weights.filter(pl.col("end_time") == first_time)["weight"]
        assert first_weights.abs().sum() < 1e-10, "First period should have zero weights (no previous signal)"
```

- [ ] **Step 3: Remove TestBacktesterCashHandling**

Delete the entire `TestBacktesterCashHandling` class (lines 396-440). Cash tracking no longer exists in return-based mode.

- [ ] **Step 4: Update TestMissingPriceHandling**

Replace the assertion in `test_missing_price_symbol_excluded_from_holdings` (lines 494-498). The test now checks weights instead of trades:

```python
        # After bar 5, ETH should have zero weight (no price data)
        eth_weights_after_5 = result.weights.filter(
            (pl.col("symbol") == "ETH") & (pl.col("end_time") > timestamps[4] + 3600000)
        )
        # ETH weights should be 0 or absent when price data is missing
        if len(eth_weights_after_5) > 0:
            assert eth_weights_after_5["weight"].abs().sum() < 1e-10
```

- [ ] **Step 5: Update TestVectorizedBacktesterIntegration comparison tests**

The `test_vectorized_vs_original_equity_curve` test compares VectorizedBacktester with LegacyBacktester. These will now produce different results by design (return-based vs qty-based). Replace this test:

```python
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_vectorized_produces_reasonable_equity(self, sample_data):
        """VectorizedBacktester should produce a reasonable equity curve."""
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = VectorizedBacktester(
            prices=sample_data,
            signal=signal,
            transaction_cost=0.0001,
            initial_capital=10000.0,
            neutralization="market",
        )
        result = bt.run()

        # Equity should start near initial capital
        first_value = result.equity_curve["total_value"][0]
        assert abs(first_value - 10000.0) < 100.0  # within 1% on first period

        # Should have returns for each period
        assert len(result.returns) == len(result.equity_curve)
```

Replace `test_vectorized_metrics_comparable`:

```python
    def test_vectorized_metrics_complete(self, sample_data):
        """Metrics should contain all expected keys."""
        close = sample_data["close"]
        signal = close.cs_rank()

        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()

        expected_keys = {
            "total_return", "annual_return", "annual_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "var_95", "cvar_95", "win_rate", "profit_factor",
        }
        assert expected_keys.issubset(result.metrics.keys())
```

- [ ] **Step 6: Update test_backtest_mask.py**

In `tests/universe/test_backtest_mask.py`, the `test_backtester_without_mask_keeps_backward_compatible_flow` test accesses `result.equity_curve` which still exists. No change needed for that test.

Verify: `cd /mnt/raid1/novis/factorium && uv run pytest tests/universe/test_backtest_mask.py -v`
Expected: All 3 tests pass (mask tests use `_prepare_data` and `_calculate_weights` directly, which still exist)

- [ ] **Step 7: Run full test suite**

Run: `cd /mnt/raid1/novis/factorium && uv run pytest tests/backtest/ tests/universe/test_backtest_mask.py -v`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add tests/backtest/test_backtester.py tests/universe/test_backtest_mask.py
git commit -m "test(backtest): update tests for return-based backtest output"
```

---

### Task 6: Update user-guide documentation

**Files:**
- Modify: `docs/user-guide/backtest.md`

- [ ] **Step 1: Update Section 3 (BacktestResult)**

Replace lines 54-83 in `docs/user-guide/backtest.md`:

````markdown
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
  - 每期每標的的最終權重（已套用約束與正規化）
- **`turnover: pl.DataFrame`**  
  - 欄位：`["end_time", "turnover", "cost"]`
  - 每期的換手率與交易成本

如需 pandas 版本，可呼叫：

```python
pandas_result = result.to_pandas()
print(pandas_result.equity_curve.tail())
print(pandas_result.metrics)
```
````

- [ ] **Step 2: Update Section 5 constraints note**

Replace the note at line 161:

```markdown
> **注意**：約束套用後會自動進行權重正規化（renormalization），確保：
> - 市場中性模式：`sum(w) = 0`，`sum(|w|) = 1`
> - Long-only 模式：`sum(w) = 1`，`w >= 0`
>
> 正規化可能導致個別權重略微超過約束上限，超出幅度與被截斷權重的佔比成正比。
```

- [ ] **Step 3: Update Section 7 workflow summary**

Replace line 196:

```markdown
5. **查看結果**：讀取 `BacktestResult.metrics`、`equity_curve`、`weights`、`turnover` 等欄位，或將結果轉成 pandas 作進一步分析。
```

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/backtest.md
git commit -m "docs(backtest): update user guide for return-based backtest output"
```

---

### Task 7: Final verification

**Files:** None (verification only)

- [ ] **Step 1: Run full project test suite**

Run: `cd /mnt/raid1/novis/factorium && uv run pytest --tb=short -q`
Expected: All tests pass

- [ ] **Step 2: Type check**

Run: `cd /mnt/raid1/novis/factorium && uv run python -m mypy src/factorium/backtest/ --ignore-missing-imports 2>&1 | head -30`
Expected: No errors (or only pre-existing ones unrelated to this change)

- [ ] **Step 3: Verify docs build**

Run: `cd /mnt/raid1/novis/factorium && uv run mkdocs build --strict 2>&1 | tail -5`
Expected: Build succeeds

- [ ] **Step 4: Quick smoke test**

Run: `cd /mnt/raid1/novis/factorium && uv run python -c "
from factorium.backtest import Backtester, BacktestResult, BacktestResultPandas
print('BacktestResult fields:', [f.name for f in BacktestResult.__dataclass_fields__.values()])
print('BacktestResultPandas importable: True')
print('Backtester is VectorizedBacktester:', Backtester.__name__)
"`
Expected:
```
BacktestResult fields: ['equity_curve', 'returns', 'weights', 'turnover', 'metrics']
BacktestResultPandas importable: True
Backtester is VectorizedBacktester: VectorizedBacktester
```
