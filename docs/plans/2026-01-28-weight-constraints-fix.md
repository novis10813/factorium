# WeightConstraint Missing Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement missing WeightConstraint features: MaxGrossExposureConstraint, MarketNeutralConstraint, and redistribution in MaxPositionConstraint.

**Architecture:** Extend the existing `WeightConstraint` system in `src/factorium/backtest/constraints.py`. Use Polars expressions for efficient weight adjustment and redistribution.

**Tech Stack:** Python, Polars, Pytest.

---

### Task 1: Add MaxGrossExposureConstraint

**Files:**
- Modify: `src/factorium/backtest/constraints.py`
- Test: `tests/backtest/test_constraints.py`

**Step 1: Write the failing test**

```python
def test_max_gross_exposure_scales_weights():
    weights = pl.DataFrame({
        "end_time": [1000] * 3,
        "symbol": ["A", "B", "C"],
        "weight": [0.6, 0.4, -0.3],  # gross = 1.3
    })
    
    constraint = MaxGrossExposureConstraint(max_exposure=1.0)
    result = constraint.apply(weights)
    
    # Should scale by 1.0/1.3
    gross = result["weight"].abs().sum()
    assert abs(gross - 1.0) < 1e-6
    # Original proportions should be maintained
    assert abs(result.filter(pl.col("symbol") == "A")["weight"][0] - 0.6 * (1.0/1.3)) < 1e-6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_constraints.py -k test_max_gross_exposure_scales_weights`
Expected: FAIL with `NameError: name 'MaxGrossExposureConstraint' is not defined`

**Step 3: Write minimal implementation**

```python
class MaxGrossExposureConstraint(WeightConstraint):
    """
    Limit total gross exposure (sum of absolute weights).
    
    If gross exposure exceeds limit, scales all weights proportionally.
    
    Args:
        max_exposure: Maximum sum(|weight|), e.g., 1.0 for 100%
    """
    
    def __init__(self, max_exposure: float):
        if max_exposure <= 0:
            raise ValueError("max_exposure must be positive")
        self.max_exposure = max_exposure
    
    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        """Scale weights if gross exposure exceeds limit."""
        # Calculate gross exposure per timestamp
        gross = weights.group_by("end_time").agg([
            pl.col("weight").abs().sum().alias("gross_exposure")
        ])
        
        # Join back to get gross per row
        weights = weights.join(gross, on="end_time", how="left")
        
        # Scale if exceeds limit
        weights = weights.with_columns([
            pl.when(pl.col("gross_exposure") > self.max_exposure)
            .then(pl.col("weight") * (self.max_exposure / pl.col("gross_exposure")))
            .otherwise(pl.col("weight"))
            .alias("weight")
        ])
        
        return weights.drop("gross_exposure")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_constraints.py -k test_max_gross_exposure_scales_weights`
Expected: PASS

**Step 5: Commit**

```bash
git add src/factorium/backtest/constraints.py tests/backtest/test_constraints.py
git commit -m "feat: add MaxGrossExposureConstraint"
```

---

### Task 2: Add MarketNeutralConstraint

**Files:**
- Modify: `src/factorium/backtest/constraints.py`
- Test: `tests/backtest/test_constraints.py`

**Step 1: Write the failing test**

```python
def test_market_neutral_makes_zero_sum():
    weights = pl.DataFrame({
        "end_time": [1000] * 3,
        "symbol": ["A", "B", "C"],
        "weight": [0.5, 0.3, 0.2],  # sum = 1.0
    })
    
    constraint = MarketNeutralConstraint()
    result = constraint.apply(weights)
    
    # Should sum to 0
    assert abs(result["weight"].sum()) < 1e-10
    # Values should be adjusted by -1/3
    assert abs(result.filter(pl.col("symbol") == "A")["weight"][0] - (0.5 - 1.0/3.0)) < 1e-10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_constraints.py -k test_market_neutral_makes_zero_sum`
Expected: FAIL with `NameError: name 'MarketNeutralConstraint' is not defined`

**Step 3: Write minimal implementation**

```python
class MarketNeutralConstraint(WeightConstraint):
    """
    Enforce market neutral: sum of weights = 0.
    
    Adjusts weights by subtracting the mean to ensure zero net exposure.
    """
    
    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        """Subtract mean weight to make market neutral."""
        # Calculate mean per timestamp
        means = weights.group_by("end_time").agg([
            pl.col("weight").mean().alias("mean_weight")
        ])
        
        # Join and subtract mean
        weights = weights.join(means, on="end_time", how="left")
        
        weights = weights.with_columns([
            (pl.col("weight") - pl.col("mean_weight")).alias("weight")
        ])
        
        return weights.drop("mean_weight")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_constraints.py -k test_market_neutral_makes_zero_sum`
Expected: PASS

**Step 5: Commit**

```bash
git add src/factorium/backtest/constraints.py tests/backtest/test_constraints.py
git commit -m "feat: add MarketNeutralConstraint"
```

---

### Task 3: Fix MaxPositionConstraint to support redistribution

**Files:**
- Modify: `src/factorium/backtest/constraints.py`
- Test: `tests/backtest/test_constraints.py`

**Step 1: Write the failing tests**

```python
def test_max_position_redistributes():
    weights = pl.DataFrame({
        "end_time": [1000] * 3,
        "symbol": ["A", "B", "C"],
        "weight": [0.6, 0.3, 0.1],  # A exceeds 0.5
    })
    
    constraint = MaxPositionConstraint(max_weight=0.5, redistribute=True)
    result = constraint.apply(weights)
    
    # A should be clipped to 0.5
    assert result.filter(pl.col("symbol") == "A")["weight"][0] == 0.5
    # Total should still be ~1.0 (redistributed)
    assert abs(result["weight"].sum() - 1.0) < 1e-10
    # The 0.1 excess should be distributed to B and C (weights 0.3 and 0.1)
    # Total of other weights = 0.3 + 0.1 = 0.4
    # B: 0.3 + 0.1 * (0.3 / 0.4) = 0.375
    # C: 0.1 + 0.1 * (0.1 / 0.4) = 0.125
    assert abs(result.filter(pl.col("symbol") == "B")["weight"][0] - 0.375) < 1e-10
    assert abs(result.filter(pl.col("symbol") == "C")["weight"][0] - 0.125) < 1e-10

def test_max_position_no_redistribute_matches_original():
    weights = pl.DataFrame({
        "end_time": [1000] * 3,
        "symbol": ["A", "B", "C"],
        "weight": [0.6, 0.3, 0.1],
    })
    
    constraint = MaxPositionConstraint(max_weight=0.5, redistribute=False)
    result = constraint.apply(weights)
    
    assert result["weight"].to_list() == [0.5, 0.3, 0.1]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/backtest/test_constraints.py -k test_max_position_redistributes`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'redistribute'`

**Step 3: Write minimal implementation**

```python
class MaxPositionConstraint(WeightConstraint):
    """
    Limit max absolute weight per position with redistribution.
    
    If any weight exceeds limit, clips it and redistributes excess
    to other positions proportionally.
    
    Args:
        max_weight: Maximum absolute weight per position
        redistribute: If True, redistribute excess to other positions
    """
    
    def __init__(self, max_weight: float, redistribute: bool = True):
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")
        self.max_weight = max_weight
        self.redistribute = redistribute
    
    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        """Clip and optionally redistribute weights."""
        if not self.redistribute:
            # Simple clipping (original behavior)
            return weights.with_columns([
                pl.when(pl.col("weight") > self.max_weight)
                .then(pl.lit(self.max_weight))
                .when(pl.col("weight") < -self.max_weight)
                .then(pl.lit(-self.max_weight))
                .otherwise(pl.col("weight"))
                .alias("weight")
            ])
        
        # Clip and track excess
        weights = weights.with_columns([
            pl.col("weight").alias("original_weight"),
            pl.when(pl.col("weight") > self.max_weight)
            .then(pl.lit(self.max_weight))
            .when(pl.col("weight") < -self.max_weight)
            .then(pl.lit(-self.max_weight))
            .otherwise(pl.col("weight"))
            .alias("clipped_weight"),
        ])
        
        # Calculate excess per timestamp
        weights = weights.with_columns([
            (pl.col("original_weight") - pl.col("clipped_weight")).alias("excess")
        ])
        
        # Calculate total excess per timestamp
        excess_totals = weights.group_by("end_time").agg([
            pl.col("excess").sum().alias("total_excess")
        ])
        
        weights = weights.join(excess_totals, on="end_time", how="left")
        
        # Redistribute excess proportionally to non-clipped positions
        # Note: If all positions are clipped, redistribution might fail or result in clipping again.
        
        weights = weights.with_columns([
            pl.when((pl.col("total_excess") != 0) & (pl.col("excess") == 0))
            .then(
                pl.col("clipped_weight") + 
                (pl.col("total_excess") * 
                 pl.col("clipped_weight").abs() / pl.col("clipped_weight").abs().filter(pl.col("excess") == 0).sum().over("end_time"))
            )
            .otherwise(pl.col("clipped_weight"))
            .fill_nan(pl.col("clipped_weight"))
            .alias("weight")
        ])
        
        return weights.select(["end_time", "symbol", "weight"])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/backtest/test_constraints.py -k test_max_position_redistributes`
Expected: PASS

**Step 5: Commit**

```bash
git add src/factorium/backtest/constraints.py tests/backtest/test_constraints.py
git commit -m "fix: update MaxPositionConstraint with redistribution"
```

---

### Task 4: Export and Final Cleanup

**Files:**
- Modify: `src/factorium/backtest/__init__.py`

**Step 1: Export new constraints**

```python
# In src/factorium/backtest/__init__.py
from .constraints import (
    WeightConstraint,
    MaxPositionConstraint,
    LongOnlyConstraint,
    MaxGrossExposureConstraint,
    MarketNeutralConstraint,
)

# Update __all__
```

**Step 2: Run all tests**

Run: `pytest tests/backtest/test_constraints.py`
Expected: PASS

**Step 3: Commit**

```bash
git add src/factorium/backtest/__init__.py
git commit -m "feat: export new constraints in backtest module"
```
