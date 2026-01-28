# Implement Additional Metrics (Sortino, Calmar, Win Rate) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Sortino ratio, Calmar ratio, and win rate to the `VectorizedBacktester` metrics.

**Architecture:** Update the `_calculate_metrics` method in `VectorizedBacktester` to include the new metrics. Ensure fallback values are handled correctly when data is insufficient.

**Tech Stack:** Python, Polars, Pandas, NumPy, Pytest

---

### Task 1: Update Tests

**Files:**
- Modify: `tests/backtest/test_vectorized.py`

**Step 1: Add TestMetricsCalculation class to tests/backtest/test_vectorized.py**

```python
class TestMetricsCalculation:
    """Tests for metrics calculation."""

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

    def test_metrics_include_win_rate(self, sample_data):
        """Metrics should include win rate."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        assert "win_rate" in result.metrics

    def test_max_drawdown_non_positive(self, sample_data):
        """Max drawdown should be <= 0."""
        signal = sample_data["close"].cs_rank()
        bt = VectorizedBacktester(prices=sample_data, signal=signal)
        result = bt.run()
        
        assert result.metrics["max_drawdown"] <= 0
```

**Step 2: Run tests to verify they pass (or fail if fallback is triggered)**

Run: `uv run pytest tests/backtest/test_vectorized.py::TestMetricsCalculation -v`
Expected: PASS (since metrics.py already has them, but we want to be sure)

---

### Task 2: Update _calculate_metrics in vectorized.py

**Files:**
- Modify: `src/factorium/backtest/vectorized.py`

**Step 1: Update the fallback dictionary in _calculate_metrics**

**Step 2: Ensure consistency with implementation requirements**

Implementation requirements from user:
- Sortino ratio: `annual_return / downside_std`
- Calmar ratio: `annual_return / abs(max_drawdown)`
- Win rate: `(returns > 0).sum() / len(returns)`

Since `calculate_metrics` already implements these, we just need to make sure the fallback returns them with `0.0`.

**Step 3: Verify tests pass**

Run: `uv run pytest tests/backtest/test_vectorized.py::TestMetricsCalculation -v`
Expected: PASS

---

### Task 3: Commit Changes

**Step 1: Commit**

```bash
git add src/factorium/backtest/vectorized.py tests/backtest/test_vectorized.py
git commit -m "feat(backtest): add Sortino, Calmar, and win rate metrics to VectorizedBacktester"
```
