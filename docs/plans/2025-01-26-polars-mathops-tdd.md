# Polars MathOpsMixin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan.

**Goal:** Reimplement MathOpsMixin using pure Polars LazyFrame expressions to support Polars-backed Factor objects.

**Architecture:** 
- Replace pandas-based implementations with Polars `pl.Expr` operations
- Maintain backward compatibility with pandas via `_data` property
- Support both factor-factor and factor-scalar operations
- Use Polars lazy evaluation with `.over()` for grouping and conditional joins

**Tech Stack:** Polars LazyFrame, pl.Expr, pl.when/then/otherwise

---

## Implementation Strategy

### Current Status
- 27 tests PASS (simple ops like abs, sign, sqrt work with current implementation)
- 7 tests FAIL:
  - `log()` and `ln()`: Use `self.data.copy()` which fails on pl.DataFrame
  - `where()`: Uses pandas merge and boolean logic
  - `div()`: Uses np.isinf() on Polars result

### Root Cause
Methods in `MathOpsMixin` are written for pandas but called on Polars LazyFrames. When `self.data` returns `pl.DataFrame`, pandas-specific methods fail.

### Solution Approach
1. **Factor API Detection**: Check if `self._lf` is a LazyFrame (Polars) or not
2. **Dual Path Implementation**: 
   - Keep pandas path working (for backward compatibility)
   - Add Polars-native path using `pl.Expr`
3. **Polars Patterns**:
   - Use `with_columns()` for element-wise operations
   - Use `when().then().otherwise()` for conditional logic
   - Use `join()` for factor-factor operations
   - Avoid `collect()` and `to_pandas()`

---

## Task 1: Analyze and Fix `log()` Method

**Files:**
- Modify: `src/factorium/factors/mixins/math_ops.py:28-44`
- Test: `tests/factors/test_math_ops_polars.py::TestMathOpsPolars_Log`

**Step 1: Write test to verify pandas path still works**

Run: `pytest tests/factors/test_math_ops_polars.py::TestMathOpsPolars_Log::test_log_basic_behavior_polars -xvs`
Expected: FAIL (AttributeError: 'DataFrame' object has no attribute 'copy')

**Step 2: Implement Polars path in `log()` using dual approach**

- Use `isinstance(self._lf, pl.LazyFrame)` to detect Polars
- Create new Polars path using:
  ```python
  # For Polars: use when().then().otherwise()
  self._lf.with_columns(
      pl.when(pl.col("factor") > 0)
        .then(pl.col("factor").log() if base is None else ...)
        .otherwise(None)
        .alias("factor")
  )
  ```
- Keep pandas path unchanged

**Step 3: Run test to verify it passes**

Run: `pytest tests/factors/test_math_ops_polars.py::TestMathOpsPolars_Log -xvs`
Expected: PASS

**Step 4: Commit**

```bash
git add src/factorium/factors/mixins/math_ops.py
git commit -m "fix(log): add Polars LazyFrame support with pl.Expr"
```

---

## Task 2: Implement `where()` Method with Polars Support

**Files:**
- Modify: `src/factorium/factors/mixins/math_ops.py:117-136`
- Test: `tests/factors/test_math_ops_polars.py::TestMathOpsPolars_Where`

**Step 1: Verify pandas path works, Polars path fails**

Run: `pytest tests/factors/test_math_ops_polars.py::TestMathOpsPolars_Where -xvs`
Expected: FAIL

**Step 2: Implement Polars `where()` using join + when/then/otherwise**

```python
if isinstance(self._lf, pl.LazyFrame):
    # Join condition factor
    cond_lf = cond._lf
    result = self._lf.join(cond_lf, on=["start_time", "end_time", "symbol"], suffix="_cond")
    
    # Apply condition logic
    if isinstance(other, self.__class__):
        other_lf = other._lf
        result = result.join(other_lf, on=["start_time", "end_time", "symbol"], suffix="_other")
        result = result.with_columns(
            pl.when(pl.col("factor_cond").is_not_null() & (pl.col("factor_cond") != 0))
              .then(pl.col("factor"))
              .otherwise(pl.col("factor_other"))
              .alias("factor")
        )
    else:
        result = result.with_columns(
            pl.when(pl.col("factor_cond").is_not_null() & (pl.col("factor_cond") != 0))
              .then(pl.col("factor"))
              .otherwise(pl.lit(other))
              .alias("factor")
        )
    
    result = result.select(["start_time", "end_time", "symbol", "factor"])
```

**Step 3: Run test to verify it passes**

Run: `pytest tests/factors/test_math_ops_polars.py::TestMathOpsPolars_Where -xvs`
Expected: PASS

**Step 4: Commit**

```bash
git add src/factorium/factors/mixins/math_ops.py
git commit -m "fix(where): add Polars LazyFrame support with join and when/then"
```

---

## Task 3: Fix `div()` Method for Polars

**Files:**
- Modify: `src/factorium/factors/mixins/math_ops.py:114-115`
- Test: `tests/factors/test_math_ops_polars.py::TestMathOpsPolars_Div::test_div_zero_divisor_polars`

**Step 1: Verify current failure**

Run: `pytest tests/factors/test_math_ops_polars.py::TestMathOpsPolars_Div::test_div_zero_divisor_polars -xvs`
Expected: FAIL (TypeError on np.isinf with Polars)

**Step 2: Add Polars-aware path to `__truediv__()`**

The issue is in the base class binary_op method. Check if `_replace_inf()` needs Polars support.

**Step 3: Update `_replace_inf()` in BaseFactor**

```python
@staticmethod
def _replace_inf(value: Union[pd.Series, pl.Expr]) -> Union[pd.Series, pl.Expr]:
    if isinstance(value, pl.Expr):
        return value.replace([np.inf, -np.inf], None)
    return value.replace([np.inf, -np.inf], np.nan)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factors/test_math_ops_polars.py::TestMathOpsPolars_Div -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add src/factorium/factors/base.py src/factorium/factors/mixins/math_ops.py
git commit -m "fix(div): handle inf values correctly for Polars expressions"
```

---

## Final Verification

After all tasks complete:

```bash
uv run pytest tests/factors/test_math_ops_polars.py -q
```

Expected: All 34 tests PASS

---

## Key Design Decisions

1. **Dual-Path Approach**: Keep pandas code paths working for backward compatibility while adding Polars paths
2. **LazyFrame Throughout**: Never call `.collect()` in MathOpsMixin to maintain lazy evaluation benefits
3. **Polars Joins**: Use Polars joins instead of pandas merge for factor-factor operations
4. **Safe Null Handling**: Use `pl.when().is_not_null()` and `pl.lit(None)` for Polars nulls instead of `np.nan`
5. **Deferred Computation**: Return results as `self.__class__(..., name)` which will wrap LazyFrame

---

## Testing Strategy

- Run individual test classes first to isolate issues
- Use `-xvs` flags to stop on first failure and show full output
- Check both pandas and Polars paths produce identical results
- Verify NaN/null handling matches between implementations
