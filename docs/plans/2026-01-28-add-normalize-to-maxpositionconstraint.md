# Add Normalize to MaxPositionConstraint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional `normalize` parameter to `MaxPositionConstraint` that, when True, renormalizes weights after clipping to preserve the original total allocation sum.

**Architecture:** 
1. Update `MaxPositionConstraint.__init__` to accept `normalize: bool = False`.
2. Update `MaxPositionConstraint.apply` to:
   - Calculate and store original weights sum per timestamp if `normalize=True`.
   - Clip weights as before.
   - If `normalize=True`, renormalize the clipped weights back to the original sum.
3. Ensure backward compatibility by defaulting `normalize` to `False`.

**Tech Stack:** Python, Polars, Pytest

---

### Task 1: Add Failing Tests for Normalize Behavior

**Files:**
- Modify: `tests/backtest/test_constraints.py`

**Step 1: Write the failing tests**

```python
def test_max_position_with_normalize():
    """normalize=True should preserve weight sum."""
    weights = pl.DataFrame({
        "end_time": [1000] * 3,
        "symbol": ["A", "B", "C"],
        "weight": [0.6, 0.3, 0.1],  # sum = 1.0
    })
    
    # Without normalize
    constraint_no_norm = MaxPositionConstraint(max_weight=0.4, normalize=False)
    result_no_norm = constraint_no_norm.apply(weights)
    # A clipped to 0.4, sum now 0.8
    assert abs(result_no_norm["weight"].sum() - 0.8) < 1e-6
    
    # With normalize
    constraint_norm = MaxPositionConstraint(max_weight=0.4, normalize=True)
    result_norm = constraint_norm.apply(weights)
    # Sum should be preserved at 1.0
    assert abs(result_norm["weight"].sum() - 1.0) < 1e-6
    # But A should still not exceed max_weight significantly
    a_weight = result_norm.filter(pl.col("symbol") == "A")["weight"][0]
    # After renormalization: 0.4 * (1.0/0.8) = 0.5
    assert a_weight <= 0.51  # Allow small tolerance

def test_max_position_backward_compatible():
    """Default behavior should remain unchanged (no normalize)."""
    weights = pl.DataFrame({
        "end_time": [1000] * 2,
        "symbol": ["A", "B"],
        "weight": [0.8, 0.2],
    })
    
    constraint = MaxPositionConstraint(max_weight=0.5)  # normalize defaults to False
    result = constraint.apply(weights)
    
    # Should just clip, not normalize
    assert result.filter(pl.col("symbol") == "A")["weight"][0] == 0.5
    assert result.filter(pl.col("symbol") == "B")["weight"][0] == 0.2
    assert abs(result["weight"].sum() - 0.7) < 1e-6  # Sum reduced
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backtest/test_constraints.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'normalize'`

---

### Task 2: Implement Normalize Parameter in MaxPositionConstraint

**Files:**
- Modify: `src/factorium/backtest/constraints.py`

**Step 1: Update __init__ and apply method**

```python
class MaxPositionConstraint(WeightConstraint):
    """
    Limit maximum absolute weight per position.
    
    Args:
        max_weight: Maximum absolute weight (e.g., 0.1 for 10%)
        normalize: If True, renormalize weights after clipping to preserve sum
    
    Example:
        >>> # Simple clip (may reduce total allocation)
        >>> constraint = MaxPositionConstraint(max_weight=0.15)
        >>> 
        >>> # Clip and renormalize to maintain total allocation
        >>> constraint = MaxPositionConstraint(max_weight=0.15, normalize=True)
    """
    
    def __init__(self, max_weight: float, normalize: bool = False):
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")
        self.max_weight = max_weight
        self.normalize = normalize
    
    def apply(self, weights: pl.DataFrame) -> pl.DataFrame:
        """Clip weights and optionally renormalize."""
        # Store original sum per timestamp
        if self.normalize:
            original_sums = weights.group_by("end_time").agg(
                pl.col("weight").sum().alias("original_sum")
            )
        
        # Clip weights
        weights = weights.with_columns(
            pl.when(pl.col("weight") > self.max_weight)
            .then(pl.lit(self.max_weight))
            .when(pl.col("weight") < -self.max_weight)
            .then(pl.lit(-self.max_weight))
            .otherwise(pl.col("weight"))
            .alias("weight")
        )
        
        # Renormalize if requested
        if self.normalize:
            # Get new sum
            new_sums = weights.group_by("end_time").agg(
                pl.col("weight").sum().alias("new_sum")
            )
            
            # Join both sums
            weights = weights.join(original_sums, on="end_time")
            weights = weights.join(new_sums, on="end_time")
            
            # Scale to preserve original sum
            weights = weights.with_columns(
                pl.when(pl.col("new_sum").abs() > 1e-10)
                .then(pl.col("weight") * pl.col("original_sum") / pl.col("new_sum"))
                .otherwise(pl.col("weight"))
                .alias("weight")
            )
            
            weights = weights.drop(["original_sum", "new_sum"])
        
        return weights
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/backtest/test_constraints.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add src/factorium/backtest/constraints.py tests/backtest/test_constraints.py
git commit -m "feat: add normalize parameter to MaxPositionConstraint"
```
