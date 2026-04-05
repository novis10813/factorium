# VectorizedBacktester Integration Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add integration tests to `tests/backtest/test_backtester.py` that compare `VectorizedBacktester` (Polars) with the original `Backtester` (Pandas) to ensure consistency.

**Architecture:** Create a new test class `TestVectorizedBacktesterIntegration` that mirrors key tests from `TestBacktester`. It will run both backtesters on the same sample data and compare their results (equity curve, metrics) within a small tolerance.

**Tech Stack:** Python, pytest, Pandas, Polars, factorium

---

### Task 1: Setup TestVectorizedBacktesterIntegration

**Files:**
- Modify: `tests/backtest/test_backtester.py`

**Step 1: Import VectorizedBacktester**
Add `VectorizedBacktester` to imports.

**Step 2: Add TestVectorizedBacktesterIntegration class**
Add the class with a mirror of the `sample_data` fixture.

**Step 3: Add test_vectorized_vs_original_equity_curve**
Compare final equity between both implementations.

**Step 4: Add test_vectorized_polars_output_types**
Verify `VectorizedBacktester` returns Polars DataFrames.

**Step 5: Add test_vectorized_metrics_comparable**
Compare metrics (Sharpe, Max Drawdown, etc.).

**Step 6: Run tests and verify**
Run `uv run pytest tests/backtest/test_backtester.py::TestVectorizedBacktesterIntegration -v`

---

### Task 2: Verification and Cleanup

**Step 1: Run all backtest tests**
Run `uv run pytest tests/backtest/ -v`

**Step 2: Self-review and fix any issues**
Ensure no regressions and consistent results.

**Step 3: Commit**
Final commit with all changes.
