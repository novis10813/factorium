# Safe Operations Semantics

This document formalizes the **safe operation semantics** used throughout Factorium for
numerical safety in factor calculations and backtesting. These conventions ensure
deterministic, reproducible results and prevent silent errors from corrupting financial signals.

---

## Core Principles

1. **Strict NaN Propagation** — Any `NaN` (or `null` in Polars) in the input window
   causes the entire window result to be `NaN`/`null`.
2. **Division Safety** — Division by values within `EPSILON` of zero returns `NaN`/`null`
   instead of `inf`.
3. **Window Completeness** — Rolling operations require a full window; partial windows
   produce `NaN`/`null`.

These rules are intentionally **stricter** than the defaults in Pandas / Polars / NumPy,
which typically skip `NaN` values. In quantitative finance, silently ignoring missing data
can produce misleading signals, so Factorium opts for explicit failure.

---

## Constants

All numeric thresholds are defined in `factorium.constants`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `EPSILON` | `1e-10` | Near-zero threshold for safe division and degenerate-case detection |
| `POSITION_EPSILON` | `EPSILON` (alias) | Legacy alias used in `backtest.utils`; identical to `EPSILON` |
| `MIN_PERIODS_PER_YEAR` | `1.0` | Lower bound for `periods_per_year` in metrics |
| `MAX_PERIODS_PER_YEAR` | `525960.0` | Upper bound (minute-level data) |

```python
# factorium/constants.py
EPSILON = 1e-10
```

---

## Safe Division

The safe division pattern appears in three contexts in the codebase:

### 1. `backtest.utils.safe_divide(a, b, default=np.nan)`

A general-purpose safe division function supporting scalar, NumPy array, and Pandas Series inputs.

**Rules:**
- If `b` is `NaN` → return `default` (default: `np.nan`)
- If `|b| <= EPSILON` → return `default`
- Otherwise → return `a / b`

```python
from factorium.backtest.utils import safe_divide

safe_divide(1.0, 0.0)        # → np.nan
safe_divide(1.0, 1e-15)      # → np.nan  (within EPSILON)
safe_divide(1.0, 2.0)        # → 0.5
safe_divide(1.0, np.nan)     # → np.nan
```

**Supported input types:**

| Type of `b` | Near-zero detection | NaN detection |
|-------------|-------------------|---------------|
| Scalar (`float`, `int`) | `abs(b) <= EPSILON` | `np.isnan(b)` |
| `np.ndarray` | `np.abs(b) <= EPSILON` | `np.isnan(b)` |
| `pd.Series` | `b.abs() <= EPSILON` | `b.isna()` |

### 2. Factor Division (`Factor.__truediv__`)

The `Factor / Factor` and `Factor / scalar` operations use Polars expressions with the
same EPSILON threshold:

```python
# Polars path (Factor / Factor)
pl.when(pl.col("other").abs() <= EPSILON)
  .then(pl.lit(None))       # → null (Polars equivalent of NaN)
  .otherwise(pl.col("factor") / pl.col("other"))

# Polars path (Factor / scalar)
pl.when(pl.lit(other).abs() <= EPSILON)
  .then(pl.lit(None))
  .otherwise(pl.col("factor") / pl.lit(other))
```

**Key difference:** The Polars path returns `null` (not `NaN`) for near-zero denominators.
This is consistent with Polars conventions where `null` represents missing data.

### 3. `MathOpsMixin.inverse()`

```python
# 1 / factor with safe division
pl.when(pl.col("factor").abs() <= EPSILON)
  .then(pl.lit(None))
  .otherwise(1 / pl.col("factor"))
```

---

## Strict NaN Propagation in Rolling Operations

All time-series operations (`ts_*`) follow **strict NaN propagation**: if any value within
the rolling window is `NaN`/`null`, or if the window is not full, the result is `NaN`/`null`.

### Mechanism

Polars rolling functions control this via the `min_samples` parameter:

```python
# min_samples=window ensures NaN if window is not full
pl.col("factor").rolling_mean(window_size=window, min_samples=window).over("symbol")
```

When `min_samples == window_size`, Polars will return `null` if:
- The window has fewer than `window` non-null values
- Any value in the window is `null`

### Operations Using This Pattern

| Operation | Polars Function | EPSILON Check |
|-----------|----------------|---------------|
| `ts_mean` | `rolling_mean(min_samples=window)` | No |
| `ts_std` | `rolling_std(min_samples=window)` | No |
| `ts_sum` | `rolling_sum(min_samples=window)` | No |
| `ts_min` | `rolling_min(min_samples=window)` | No |
| `ts_max` | `rolling_max(min_samples=window)` | No |
| `ts_median` | `rolling_median(min_samples=window)` | No |
| `ts_kurtosis` | `rolling_kurtosis(min_samples=window)` | No |
| `ts_skewness` | `rolling_skew(min_samples=window)` | No |
| `ts_rank` | `rolling_rank(min_samples=window)` | Yes (constant std check) |
| `ts_scale` | min/max + division | Yes (range < EPSILON) |
| `ts_zscore` | mean/std + division | Yes (std < EPSILON) |
| `ts_corr` | manual cov / (std_x × std_y) | Yes (either std < EPSILON) |
| `ts_beta` | manual cov / var | Yes (var < EPSILON) |
| `ts_cv` | std / \|mean\| | Yes (adds 1e-10 bias term) |

### Explicit NaN-in-Window Mask

For operations requiring EPSILON checks, an explicit NaN mask is computed:

```python
nan_in_window = (
    (pl.col("factor").is_null() | pl.col("factor").is_nan())
    .cast(pl.Int64)
    .rolling_max(window_size=window, min_samples=window)
    .over("symbol")
    .fill_null(1)   # Treat incomplete windows as having NaN
)
```

This mask is `> 0` if **any** value in the window is `NaN` or `null`, or if the window is
not fully populated. Result computation then uses:

```python
pl.when(nan_in_window > 0).then(pl.lit(None)).otherwise(computed_expr)
```

---

## Strict NaN Propagation in Cross-Sectional Operations

Cross-sectional operations (`cs_*`) apply a **strict NaN mask across the entire
cross-section** at each time step:

```python
# If ANY symbol has NaN at time t, ALL symbols get NaN at time t
nan_mask = (pl.col("factor").is_null() | pl.col("factor").is_nan()).any().over("end_time")
```

### Operations Using This Pattern

| Operation | EPSILON Check | Special Handling |
|-----------|---------------|------------------|
| `cs_rank` | No | Returns rank / count |
| `cs_zscore` | No (std=0 → ±inf, but caught by NaN mask) | — |
| `cs_demean` | No | — |
| `cs_winsorize` | No | Clips to quantile bounds |
| `cs_neutralize` | Yes (var_x < EPSILON → null) | OLS regression |
| `cs_mean` / `cs_median` | No | — |

---

## Degenerate-Case Handling

Beyond NaN propagation and division safety, specific operations have additional
degenerate-case guards:

| Operation | Degenerate Condition | Result |
|-----------|---------------------|--------|
| `ts_rank` | `std < EPSILON` (all values identical) | `null` |
| `ts_scale` | `max - min <= EPSILON` (no range) | `null` |
| `ts_zscore` | `std <= EPSILON` (no variance) | `null` |
| `ts_corr` | `std_x <= EPSILON` or `std_y <= EPSILON` | `null` |
| `ts_beta` | `var_x <= EPSILON` | `null` |
| `cs_neutralize` | `var_x <= EPSILON` | `null` |
| `cs_neutralize` (engine) | `std(x) < EPSILON` | `NaN` (NumPy path) |
| `inverse()` | `|factor| <= EPSILON` | `null` |
| `log()` | `factor <= 0` | `null` |
| `sqrt()` | `factor <= 0` | `null` |

---

## Edge Cases

### Empty Data

- `safe_divide` with empty `pd.Series` → empty `pd.Series`
- `neutralize_weights` with empty input → empty `pd.Series(dtype=float)`
- Factor operations on empty DataFrames → empty result DataFrame

### All NaN Input

- Rolling operations → all `null` output (no valid windows)
- Cross-sectional operations → all `null` output (NaN mask activates)

### Single-Element Window

- `ts_std(window=1)` → always `0.0` (single-value std is 0)
- `ts_corr(window=1)` → all `null` (needs window >= 2)
- `ts_beta(window=1)` → all `null` (needs window >= 2)

### Infinity Handling

Some operations explicitly replace `inf` / `-inf` with `null`:

```python
# ts_scale, ts_zscore
z_expr = pl.when(z_expr.is_finite()).then(z_expr).otherwise(pl.lit(None))

# ts_jumpiness, ts_vr
result._lf = result._lf.with_columns(
    pl.col("factor").replace(float("inf"), None).replace(float("-inf"), None)
)
```

---

## Summary Table

| Pattern | Where Used | Threshold | Missing Value |
|---------|-----------|-----------|---------------|
| Safe division | `safe_divide`, `__truediv__`, `inverse` | `EPSILON` (1e-10) | `NaN` (Pandas/NumPy) / `null` (Polars) |
| Strict NaN propagation (rolling) | All `ts_*` operations | `min_samples=window` | `null` |
| Strict NaN propagation (cross-section) | All `cs_*` operations | `.any().over("end_time")` | `null` |
| Variance/std guard | `ts_corr`, `ts_beta`, `ts_rank`, `cs_neutralize` | `EPSILON` | `null` |
| Range guard | `ts_scale` | `EPSILON` | `null` |
| Infinity filter | `ts_zscore`, `ts_scale`, `ts_jumpiness`, `ts_vr` | `is_finite()` | `null` |
