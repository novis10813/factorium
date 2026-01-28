# FactorAnalyzer Polars Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `FactorAnalyzer` to use Polars for data preparation and analysis, improving performance and avoiding OOM issues.

**Architecture:** Use Polars LazyFrame for joining factor and price data and calculating forward returns. Keep `_clean_data` as a Polars DataFrame. Update analysis methods to use Polars vectorized operations, only converting to Pandas for final results.

**Tech Stack:** Polars, Pandas

---

### Task 1: Update `prepare_data` to use Polars

**Files:**
- Modify: `src/factorium/factors/analyzer.py`

**Step 1: Implement Polars-based `prepare_data`**
Update `prepare_data` to use Polars LazyFrame for joining and return calculation.

**Step 2: Update `__init__` if necessary**
Ensure `self.prices` is handled correctly.

**Step 3: Run existing tests**
Run `uv run pytest tests/factors/test_analyzer.py -v`
They might fail because they expect `pd.DataFrame` from `prepare_data`. I may need to update tests or keep `prepare_data` returning Pandas for compatibility if many things depend on it, but the goal is Polars. Actually, I'll update the tests to handle either, or change them to expect Polars.

### Task 2: Update `calculate_ic` and `calculate_ic_summary` to use Polars

**Files:**
- Modify: `src/factorium/factors/analyzer.py`

**Step 1: Implement `calculate_ic` using Polars**
Use Polars `group_by("start_time").agg` with `pl.corr(..., method="spearman")`.

**Step 2: Implement `calculate_ic_summary` using Polars**
Calculate mean, std, t-stat, and IR using Polars aggregations.

### Task 3: Update `calculate_quantile_returns` and `calculate_cumulative_returns` to use Polars

**Files:**
- Modify: `src/factorium/factors/analyzer.py`

**Step 1: Implement `calculate_quantile_returns` using Polars**
Use Polars for quantile assignment and grouping.

**Step 2: Implement `calculate_cumulative_returns` using Polars**
Use Polars for cumulative product calculation.

### Task 4: Verification and Performance Test

**Files:**
- Create: `tests/factors/test_analyzer_performance.py`

**Step 1: Add a test to verify no Pandas conversion in hot path**
Use mocking to ensure `to_pandas()` is not called during `prepare_data`.

**Step 2: Run all tests**
Ensure everything passes.
