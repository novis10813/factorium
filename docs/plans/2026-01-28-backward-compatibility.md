# Backward Compatibility Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add backward compatibility so existing code using the old `Backtester` can use `VectorizedBacktester` with minimal changes, and ensure it is properly exported.

**Architecture:** Update the `backtest` module's `__init__.py` to export `VectorizedBacktester` and add compatibility tests for the `to_pandas()` method of `VectorizedBacktester`'s result.

**Tech Stack:** Python, Polars, Pandas, Pytest

---

### Task 1: Update backtest module exports

**Files:**
- Modify: `src/factorium/backtest/__init__.py`

**Step 1: Modify imports and __all__**

Update `src/factorium/backtest/__init__.py` to match the requested structure, specifically removing the `VectorizedBacktestResult` alias and export if it's not requested.

```python
from .backtester import Backtester, BacktestResult
from .metrics import calculate_metrics
from .portfolio import Portfolio
from .vectorized import VectorizedBacktester
from .utils import (
    MAX_PERIODS_PER_YEAR,
    MIN_PERIODS_PER_YEAR,
    POSITION_EPSILON,
    frequency_to_periods_per_year,
    neutralize_weights,
    normalize_weights,
    parse_frequency_to_seconds,
)

__all__ = [
    "Backtester",
    "BacktestResult",
    "VectorizedBacktester",
    "Portfolio",
    "calculate_metrics",
    "frequency_to_periods_per_year",
    "neutralize_weights",
    "normalize_weights",
    "parse_frequency_to_seconds",
    "POSITION_EPSILON",
    "MIN_PERIODS_PER_YEAR",
    "MAX_PERIODS_PER_YEAR",
]
```

**Step 2: Verify exports**

Run: `python -c "from factorium.backtest import VectorizedBacktester; print('Success')"`
Expected: "Success"

### Task 2: Add backward compatibility tests

**Files:**
- Modify: `tests/backtest/test_vectorized.py`

**Step 1: Add TestBackwardCompatibility class**

Append the following class to `tests/backtest/test_vectorized.py`:

```python
class TestBackwardCompatibility:
    """Tests for backward compatibility with existing Backtester API."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 20, 3600000))
        
        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH"]:
                base_price = {"BTC": 100.0, "ETH": 50.0}[symbol]
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

    def test_can_import_from_backtest_module(self):
        """Should be able to import VectorizedBacktester from backtest."""
        from factorium.backtest import VectorizedBacktester
        assert VectorizedBacktester is not None

    def test_result_to_pandas_compatibility(self, sample_data):
        """BacktestResult.to_pandas() should return pandas DataFrames."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        pandas_result = result.to_pandas()
        
        import pandas as pd
        assert isinstance(pandas_result.equity_curve, pd.DataFrame)
        assert isinstance(pandas_result.returns, pd.DataFrame)
        assert isinstance(pandas_result.metrics, dict)
```

**Step 2: Run the new tests**

Run: `uv run pytest tests/backtest/test_vectorized.py::TestBackwardCompatibility -v`
Expected: All tests pass.

**Step 3: Run all backtest tests**

Run: `uv run pytest tests/backtest/ -v`
Expected: All tests pass.

### Task 3: Commit changes

**Step 1: Commit**

Run:
```bash
git add src/factorium/backtest/__init__.py tests/backtest/test_vectorized.py
git commit -m "feat(backtest): add backward compatibility for VectorizedBacktester"
```
