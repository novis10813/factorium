# Update Documentation to Match Actual API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update project documentation to reflect the actual implemented API, new features, and provide a migration guide for breaking changes.

**Architecture:** 
- Update implementation status in the research engine plan.
- Update user guides with new examples and API details.
- Create a new migration guide for v2.0.

**Tech Stack:** Markdown

---

### Task 1: Update Implementation Status in 003 Plan

**Files:**
- Modify: `docs/plans/003_research_engine_implementation.md`

**Step 1: Add implementation status section at the top**

Add the following section after the header:

```markdown
## Implementation Status (2026-01-28)

All tasks from this plan have been implemented. Some implementation details 
differ slightly from the original pseudo-code:

### Key Implementation Notes:

1. **FactorAnalysisResult**: `analyze()` returns a structured `FactorAnalysisResult` 
   dataclass instead of a plain dict. Use `.to_dict()` for backward compatibility.

2. **MaxPositionConstraint**: Added `normalize=True` option to preserve weight sum 
   after clipping.

3. **WeightConstraint Strategy Pattern**: Constraints are implemented as separate 
   classes (Strategy pattern) rather than a single configuration object.

4. **safe_divide**: Updated to use `EPSILON` (1e-10) threshold per AGENTS.md.

5. **Backtester Alias**: `Backtester` now points to `VectorizedBacktester`. 
   Use `LegacyBacktester` for the old iterative implementation.
```

**Step 2: Commit**

```bash
git add docs/plans/003_research_engine_implementation.md
git commit -m "docs: update implementation status in 003 plan"
```

---

### Task 2: Update Backtest User Guide

**Files:**
- Modify: `docs/user-guide/backtest.md`

**Step 1: Add VectorizedBacktester and Constraints examples**

Read `docs/user-guide/backtest.md` and append or integrate the following:

```markdown
## Vectorized Backtesting

The default `Backtester` is now vectorized using Polars, providing significant performance gains for large datasets.

```python
from factorium.backtest import Backtester

bt = Backtester(
    prices=agg,
    signal=signal,
    initial_capital=10000.0,
    neutralization="market"
)
result = bt.run()
```

## Constraints

You can add constraints to the backtest to enforce position limits or other requirements.

```python
from factorium.backtest import MaxPositionConstraint

# Limit any single position to 10% weight, and normalize remaining weights
# to preserve the total weight sum (e.g., 0.0 for market neutral)
constraint = MaxPositionConstraint(max_weight=0.1, normalize=True)

bt = Backtester(
    prices=agg,
    signal=signal,
    constraints=[constraint]
)
```

## Working with Results

`BacktestResult` uses Polars internally but provides a pandas compatibility layer.

```python
result = bt.run()

# Access Polars DataFrames
result.equity_curve  # pl.DataFrame

# Convert to pandas for analysis or plotting
pandas_result = result.to_pandas()
pandas_result.equity_curve.plot()
```
```

**Step 2: Commit**

```bash
git add docs/user-guide/backtest.md
git commit -m "docs: update backtest user guide with vectorized and constraints examples"
```

---

### Task 3: Update Factor User Guide

**Files:**
- Modify: `docs/user-guide/factor.md`

**Step 1: Add FactorAnalysisResult, CompositeFactor, and ResearchSession examples**

Append or integrate the following into `docs/user-guide/factor.md`:

```markdown
## Factor Analysis

`FactorAnalyzer.analyze()` returns a `FactorAnalysisResult` dataclass.

```python
analyzer = session.analyze(factor)
result = analyzer.analyze()

print(f"Mean IC: {result.ic_summary['mean_ic']}")
result.plot_ic_distribution()
```

## Composite Factors

Combine multiple factors easily using `CompositeFactor`.

```python
from factorium.factors import CompositeFactor

# Create a composite factor by averaging Z-scores of multiple factors
composite = CompositeFactor.from_zscore([mom_factor, vol_factor])
combined = composite.to_factor()
```

## Research Workflow

Use `ResearchSession` to manage your entire research pipeline.

```python
from factorium import ResearchSession

# Load data
session = ResearchSession.load("data.parquet")

# Create factors using expressions
signal = session.create_factor("ts_mean(close, 20)", name="momentum")

# Analyze
result = session.analyze(signal)
print(result.ic_summary)

# Quick Report
report = session.quick_report(signal)
```
```

**Step 2: Commit**

```bash
git add docs/user-guide/factor.md
git commit -m "docs: update factor user guide with session and composite examples"
```

---

### Task 4: Create Migration Guide

**Files:**
- Create: `docs/dev/migration-guide.md`

**Step 1: Write migration guide content**

```markdown
# Migration Guide: v1.x to v2.0

## Breaking Changes

### 1. Backtester Default Changed

The default `Backtester` is now `VectorizedBacktester` (Polars-based).

**Before:**
```python
from factorium.backtest import Backtester
bt = Backtester(prices, signal)  # Old iterative implementation
```

**After:**
```python
from factorium.backtest import Backtester  # Now VectorizedBacktester
bt = Backtester(prices, signal)
result = bt.run()

# For old behavior:
from factorium.backtest import LegacyBacktester
```

### 2. BacktestResult Returns Polars DataFrames

**Before:**
```python
result.equity_curve  # pd.Series with DatetimeIndex
```

**After:**
```python
result.equity_curve  # pl.DataFrame with end_time column

# For pandas compatibility:
pandas_result = result.to_pandas()
pandas_result.equity_curve  # pd.DataFrame
```

### 3. FactorAnalyzer.analyze() Returns Dataclass

**Before:**
```python
result = analyzer.analyze()  # dict
ic_mean = result["ic_summary"]["mean_ic"]
```

**After:**
```python
result = analyzer.analyze()  # FactorAnalysisResult
ic_mean = result.ic_summary["mean_ic"]

# For dict compatibility:
result_dict = result.to_dict()
```

### 4. safe_divide Uses EPSILON Threshold

Division by values within EPSILON (1e-10) of zero now returns NaN.

## New Features

### Constraints with normalize

```python
from factorium.backtest import MaxPositionConstraint

# Preserve weight sum after clipping
constraint = MaxPositionConstraint(max_weight=0.1, normalize=True)
```

### ResearchSession Workflow

```python
from factorium import ResearchSession

session = ResearchSession.load("data.parquet")
signal = session.create_factor("ts_mean(close, 20)", "momentum")
result = session.analyze(signal)  # FactorAnalysisResult
report = session.quick_report(signal)
print(report)
```

### CompositeFactor

```python
from factorium.factors import CompositeFactor

composite = CompositeFactor.from_zscore([factor1, factor2])
combined = composite.to_factor()
```
```

**Step 2: Commit**

```bash
git add docs/dev/migration-guide.md
git commit -m "docs: add migration guide for v2.0"
```
