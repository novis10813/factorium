# WeightConstraint Missing Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `MaxGrossExposureConstraint` and `MarketNeutralConstraint` to the backtest module.

**Architecture:** Implement two new subclasses of `WeightConstraint` in `constraints.py`, export them in `__init__.py`, and add unit tests.

**Tech Stack:** Python, Polars, Pytest

---

### Task 1: Add constraint classes to `constraints.py`

**Files:**
- Modify: `src/factorium/backtest/constraints.py`

**Step 1: Write minimal implementation**

Add the following classes to `src/factorium/backtest/constraints.py`:

```python
class MaxGrossExposureConstraint(WeightConstraint):
    """Limit sum(|weights|) per timestamp."""
    
    def __init__(self, max_exposure: float):
        if max_exposure <= 0:
            raise ValueError("max_exposure must be positive")
        self.max_exposure = max_exposure
    
    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        # Group by end_time, calculate gross, scale if needed
        gross = weights.group_by("end_time").agg(
            pl.col("weight").abs().sum().alias("gross")
        )
        
        weights = weights.join(gross, on="end_time")
        
        weights = weights.with_columns(
            pl.when(pl.col("gross") > self.max_exposure)
            .then(pl.col("weight") * self.max_exposure / pl.col("gross"))
            .otherwise(pl.col("weight"))
            .alias("weight")
        )
        
        return weights.drop("gross")


class MarketNeutralConstraint(WeightConstraint):
    """Enforce sum(weights) = 0 per timestamp."""
    
    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        # Calculate mean per timestamp
        means = weights.group_by("end_time").agg(
            pl.col("weight").mean().alias("mean_w")
        )
        
        weights = weights.join(means, on="end_time")
        
        weights = weights.with_columns(
            (pl.col("weight") - pl.col("mean_w")).alias("weight")
        )
        
        return weights.drop("mean_w")
```

**Step 2: Verify syntax**

Run: `python -m py_compile src/factorium/backtest/constraints.py`
Expected: Success

---

### Task 2: Update exports in `__init__.py`

**Files:**
- Modify: `src/factorium/backtest/__init__.py`

**Step 1: Update imports and `__all__`**

```python
from .constraints import (
    WeightConstraint,
    MaxPositionConstraint,
    LongOnlyConstraint,
    MaxGrossExposureConstraint,
    MarketNeutralConstraint,
)

__all__ = [
    ...,
    "MaxGrossExposureConstraint",
    "MarketNeutralConstraint",
]
```

---

### Task 3: Add tests to `test_constraints.py`

**Files:**
- Modify: `tests/backtest/test_constraints.py`

**Step 1: Add test functions**

Add the following tests to `tests/backtest/test_constraints.py`:

```python
def test_max_gross_exposure():
    weights = pl.DataFrame({
        "end_time": [1000] * 3,
        "symbol": ["A", "B", "C"],
        "weight": [0.6, 0.5, -0.3],  # gross = 1.4
    })
    
    constraint = MaxGrossExposureConstraint(max_exposure=1.0)
    result = constraint.apply(weights)
    
    gross = result["weight"].abs().sum()
    assert abs(gross - 1.0) < 1e-6

def test_market_neutral():
    weights = pl.DataFrame({
        "end_time": [1000] * 2,
        "symbol": ["A", "B"],
        "weight": [0.6, 0.4],
    })
    
    constraint = MarketNeutralConstraint()
    result = constraint.apply(weights)
    
    assert abs(result["weight"].sum()) < 1e-10
```

**Step 2: Run tests**

Run: `uv run pytest tests/backtest/test_constraints.py -v`
Expected: All tests pass (including existing ones)

---

### Task 4: Final Verification and Commit

**Step 1: Self-review and final test run**

Run: `uv run pytest tests/backtest/test_constraints.py -v`

**Step 2: Commit changes**

```bash
git add src/factorium/backtest/constraints.py src/factorium/backtest/__init__.py tests/backtest/test_constraints.py
git commit -m "feat: add MaxGrossExposureConstraint and MarketNeutralConstraint"
```
