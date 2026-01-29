# CompositeFactor Factory Methods Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `from_equal_weights`, `from_weights`, and `from_zscore` factory methods for `CompositeFactor` to simplify factor composition.

**Architecture:** Add class methods to `CompositeFactor` that handle the creation of `CompositeFactor` instances with specific configurations (equal weights, custom weights with validation, or z-score normalization).

**Tech Stack:** Python, Polars, Numpy, Pytest.

---

### Task 1: Implement `from_equal_weights`

**Files:**
- Modify: `src/factorium/factors/composite.py`
- Test: `tests/factors/test_composite.py`

**Step 1: Write the failing test**

In `tests/factors/test_composite.py`, add `test_from_equal_weights`.

```python
    def test_from_equal_weights(self, sample_data):
        """Should create composite with equal weights."""
        f1 = sample_data["close"].cs_rank()
        f2 = sample_data["volume"].cs_rank()
        
        composite = CompositeFactor.from_equal_weights([f1, f2], name="equal_combo")
        
        assert composite.name == "equal_combo"
        assert len(composite.weights) == 2
        assert composite.weights[0] == composite.weights[1]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/factors/test_composite.py::TestCompositeFactor::test_from_equal_weights -v`
Expected: FAIL with `AttributeError: type object 'CompositeFactor' has no attribute 'from_equal_weights'`

**Step 3: Write minimal implementation**

In `src/factorium/factors/composite.py`, add the `from_equal_weights` class method.

```python
    @classmethod
    def from_equal_weights(cls, factors: List[Factor], name: str = "composite") -> "CompositeFactor":
        """
        Create composite with equal weights.
        
        Args:
            factors: List of factors
            name: Name for composite
        
        Returns:
            CompositeFactor with equal weights
        """
        return cls(factors, weights=None, name=name)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/factors/test_composite.py::TestCompositeFactor::test_from_equal_weights -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/factorium/factors/composite.py tests/factors/test_composite.py
git commit -m "feat(factors): add from_equal_weights factory method to CompositeFactor"
```

---

### Task 2: Implement `from_weights`

**Files:**
- Modify: `src/factorium/factors/composite.py`
- Test: `tests/factors/test_composite.py`

**Step 1: Write the failing test**

In `tests/factors/test_composite.py`, add `test_from_weights_validates_sum`.

```python
    def test_from_weights_validates_sum(self, sample_data):
        """Should validate that weights sum to 1."""
        f1 = sample_data["close"].cs_rank()
        f2 = sample_data["volume"].cs_rank()
        
        # Valid weights
        composite = CompositeFactor.from_weights([f1, f2], [0.7, 0.3])
        assert composite is not None
        
        # Invalid weights (don't sum to 1)
        with pytest.raises(ValueError, match="sum to 1"):
            CompositeFactor.from_weights([f1, f2], [0.5, 0.6])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/factors/test_composite.py::TestCompositeFactor::test_from_weights_validates_sum -v`
Expected: FAIL with `AttributeError: type object 'CompositeFactor' has no attribute 'from_weights'`

**Step 3: Write minimal implementation**

In `src/factorium/factors/composite.py`, add the `from_weights` class method.

```python
    @classmethod
    def from_weights(cls, factors: List[Factor], weights: List[float], name: str = "composite") -> "CompositeFactor":
        """
        Create composite with custom weights.
        
        Args:
            factors: List of factors
            weights: Custom weights (must sum to 1)
            name: Name for composite
        
        Returns:
            CompositeFactor with given weights
        """
        # Validate weights sum to approximately 1
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        return cls(factors, weights=weights, name=name)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/factors/test_composite.py::TestCompositeFactor::test_from_weights_validates_sum -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/factorium/factors/composite.py tests/factors/test_composite.py
git commit -m "feat(factors): add from_weights factory method to CompositeFactor"
```

---

### Task 3: Implement `from_zscore`

**Files:**
- Modify: `src/factorium/factors/composite.py`
- Test: `tests/factors/test_composite.py`

**Step 1: Write the failing test**

In `tests/factors/test_composite.py`, add `test_from_zscore_standardizes`.

```python
    def test_from_zscore_standardizes(self, sample_data):
        """Should create composite with z-score normalization."""
        f1 = sample_data["close"]
        f2 = sample_data["volume"]
        
        composite = CompositeFactor.from_zscore([f1, f2])
        result = composite.to_factor()
        
        assert result.name == "composite_zscore"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/factors/test_composite.py::TestCompositeFactor::test_from_zscore_standardizes -v`
Expected: FAIL with `AttributeError: type object 'CompositeFactor' has no attribute 'from_zscore'`

**Step 3: Write minimal implementation**

In `src/factorium/factors/composite.py`, add the `from_zscore` class method.

```python
    @classmethod
    def from_zscore(cls, factors: List[Factor], name: str = "composite_zscore") -> "CompositeFactor":
        """
        Create composite by standardizing factors first (z-score).
        
        Each factor is standardized: (x - mean) / std
        Then combined with equal weights.
        
        Args:
            factors: List of factors
            name: Name for composite
        
        Returns:
            CompositeFactor with z-score normalized factors
        """
        import numpy as np
        
        # Standardize each factor
        standardized = []
        for factor in factors:
            df = factor.lazy.collect()
            
            # Calculate mean and std per symbol (cross-sectionally)
            df = df.with_columns([
                pl.col("factor").mean().over(["start_time", "end_time"]).alias("cs_mean"),
                pl.col("factor").std().over(["start_time", "end_time"]).alias("cs_std"),
            ])
            
            # Z-score: (factor - mean) / std
            df = df.with_columns([
                ((pl.col("factor") - pl.col("cs_mean")) / pl.col("cs_std"))
                .fill_nan(0.0)  # Handle division by zero
                .alias("factor")
            ])
            
            df = df.select(["start_time", "end_time", "symbol", "factor"])
            
            from .core import Factor
            standardized.append(Factor(df, name=f"{factor.name}_zscore"))
        
        return cls(standardized, weights=None, name=name)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/factors/test_composite.py::TestCompositeFactor::test_from_zscore_standardizes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/factorium/factors/composite.py tests/factors/test_composite.py
git commit -m "feat(factors): add from_zscore factory method to CompositeFactor"
```
