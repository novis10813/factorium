# Return-Based VectorizedBacktester Redesign

**Date**: 2026-04-05
**Status**: Approved
**Breaking Change**: Yes (target 0.5.0)

## Problem

The current `VectorizedBacktester` uses fixed `initial_capital` to compute `target_qty`, simulating a constant-notional strategy rather than a fully-invested strategy. This causes position sizing to diverge from actual portfolio value over time. Additionally, `_calculate_metrics` duplicates and overrides `calculate_metrics()` with subtly different logic, and constraints are applied without renormalization.

See: `docs/plans/2026-02-14-gemini-review-fixes.md` for the full review report context.

## Decision

Replace the qty-based simulation with a **return-based backtest**. This eliminates path-dependent position tracking entirely and is the standard approach for fast factor research iteration.

## Core Calculation Flow

```
Signal --> Weights --> Constraints --> Renormalize --> Portfolio Return --> Equity Curve
```

Per-period logic:

```python
# 1. Weight calculation (cross-sectional neutralization, same as current)
w_t = neutralize(signal_{t-1})   # shift 1 to avoid lookahead

# 2. Apply constraints
w_t = apply_constraints(w_t)

# 3. Renormalization (NEW)
#    market neutral: sum(w) = 0, sum(|w|) = 1
#    long-only:      sum(w) = 1, all w >= 0

# 4. Asset returns
r_t = price_t / price_{t-1} - 1   # per symbol

# 5. Portfolio return
portfolio_return_t = sum(w_t * r_t)

# 6. Transaction cost (simple approximation)
#    buy side:  sum(max(0, w_t - w_{t-1})) * buy_rate
#    sell side: sum(max(0, w_{t-1} - w_t)) * sell_rate
cost_t = buy_cost + sell_cost
net_return_t = portfolio_return_t - cost_t

# 7. Equity curve
equity_t = equity_{t-1} * (1 + net_return_t)
```

### Edge case: first period (t=0)

- `signal_{t-1}` does not exist at t=0, so `w_0` is all zeros (no position). The first period with actual weights is t=1.
- `w_{t-1}` for turnover: at t=1, `w_0 = 0` for all symbols, so the full weight vector counts as turnover (initial entry cost).
- `r_0` is not computed (no previous price). The equity curve starts at `initial_capital` with no return.

### Transaction Cost Approximation

Uses **target weight differences** (`|w_t - w_{t-1}|`) to approximate turnover, not drift-adjusted weight differences. In reality, at the start of period t, the actual portfolio weights have drifted from `w_{t-1}` due to differential asset returns during period t-1. The true turnover would be `|w_t - w_{t-1,drifted}|`. We deliberately ignore this drift for simplicity.

Impact: turnover may be slightly over- or under-estimated depending on whether drift moves weights toward or away from the new target. For factor research iteration this is negligible.

Asymmetric fees supported: buy-side and sell-side weight changes are separated and multiplied by their respective rates.

## Output Structure

```python
@dataclass
class BacktestResult:
    equity_curve: pl.DataFrame    # [end_time, total_value]
    returns: pl.DataFrame         # [end_time, return]
    weights: pl.DataFrame         # [end_time, symbol, weight]
    turnover: pl.DataFrame        # [end_time, turnover, cost]
    metrics: dict[str, float]

@dataclass
class BacktestResultPandas:
    equity_curve: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.DataFrame
    metrics: dict[str, float]
```

Removed from current `BacktestResult`:
- `trades` — no absolute position tracking in return-based mode
- `portfolio_history` — no cash/market_value decomposition

Added:
- `weights` — final per-symbol weights after constraints + renormalization
- `turnover` — per-period turnover and cost

## Constraints & Renormalization

### Application order

```
user constraints --> renormalization (always last)
```

Renormalization is the final step, ensuring weights satisfy strategy semantics. This means constraint limits may be slightly exceeded after renormalization (e.g., `MaxPositionConstraint(0.1)` may produce weights slightly above 0.1 after renormalization to sum=1). The exceedance is proportional to the fraction of total weight that was clipped — worst case is few assets with tight constraints (e.g., 5 assets with max_weight=0.1 in long-only means all are clipped to 0.1, then renormalized back to 0.2 each). This is the standard industry trade-off; iterative projection is not worth the complexity for fast iteration.

### Renormalization logic

New function in `utils.py`:

```python
def renormalize_weights(df: pl.DataFrame, neutralization: str) -> pl.DataFrame:
    if neutralization == "market":
        # 1. demean: w = w - mean(w) over end_time
        # 2. scale:  w = w / sum(|w|) over end_time
        #    if sum(|w|) == 0 -> all weights = 0
    else:  # neutralization == "none" (long-only)
        # 1. clip negative to 0 (defensive)
        # 2. w = w / sum(w) over end_time
        #    if sum(w) == 0 -> all weights = 0
```

Parameter uses `neutralization` (not `mode`) to match `VectorizedBacktester.__init__` naming.

### MaxGrossExposureConstraint fix

Drop `gross` column before join if it already exists, preventing duplicate column issues on repeated apply.

## `summary()` Method

Current `summary()` returns `num_trades: len(self._result.trades)`. With trades removed, replace with turnover-based information:

```python
def summary(self) -> dict[str, Any]:
    return {
        "initial_capital": self.initial_capital,
        "final_value": ...,
        "total_turnover": self._result.turnover["turnover"].sum(),
        "total_cost": self._result.turnover["cost"].sum(),
        **self._result.metrics,
    }
```

`num_trades` is removed — in return-based mode there are no discrete trades, only continuous weight changes.

## Mask Functionality

The existing `mask` parameter remains unchanged. Masked symbols have their signal set to null before weight calculation, resulting in weight = 0. These zero-weight symbols do **not** participate in the renormalization denominator (sum/abs_sum), because `neutralize_weights_polars` computes mean/abs_sum over non-null signals, and masked symbols are null before the fill_null(0.0) step. Renormalization follows the same pattern — only non-zero weights contribute to the normalization factor.

## Lookahead Bias Testing

Current `test_no_lookahead_bias` uses `result.trades["end_time"]` to verify timing. With trades removed, replace with weight-based verification: confirm that `result.weights` at time t uses signals from time t-1 by checking that a known signal change at time t does not affect weights until time t+1.

## Metrics Simplification

`_calculate_metrics` in `VectorizedBacktester` will directly return the result of `calculate_metrics()` without overriding any values. All metric logic lives in `metrics.py` as the single source of truth.

## Issues Addressed

| # | Issue | Resolution |
|---|-------|------------|
| #1 | Fixed initial_capital for target_qty | Eliminated — return-based has no qty tracking |
| #2 | Duplicated/inconsistent metrics | `_calculate_metrics` delegates entirely to `calculate_metrics()` |
| #3 | No renormalization after constraints | New renormalization step added |
| #4 | trade_qty == 0 semantics | Eliminated — no trade_qty in return-based |
| #6 | full_rebalance missing | Eliminated — return-based rebalances by definition each period |
| #7 | net_buy short-selling semantics | Eliminated — no cash flow tracking |
| #8 | MaxGrossExposureConstraint duplicate join | Fixed — drop column before join |
| #12 | BacktestResultPandas not in __all__ | Added to __all__ |

## Issues NOT Addressed (out of scope)

| # | Issue | Reason |
|---|-------|--------|
| #5 | Left join dilutes factor exposure | Behavior is reasonable; document only |
| #9 | Sortino ddof choice | Documentation-level, correct behavior |
| #10 | win_rate zero-return handling | Documentation-level |
| #11 | Missing plot_equity | Nice-to-have, separate task |

## Files to Modify

| File | Changes |
|------|---------|
| `vectorized.py` | Rewrite `_calculate_positions` → `_calculate_returns`, rewrite `_calculate_equity`, add asset return computation in `_prepare_data`, simplify `_calculate_metrics`, update `summary()`, update `BacktestResult`/`BacktestResultPandas`/`to_pandas()` |
| `utils.py` | Add `renormalize_weights()` function |
| `constraints.py` | Fix `MaxGrossExposureConstraint.apply()` duplicate join |
| `__init__.py` | Add `BacktestResultPandas` to `__all__` |
| `tests/` | Update all VectorizedBacktester tests for new output structure; rewrite lookahead bias test to use weights |
| `docs/user-guide/backtest.md` | Update output structure documentation (remove trades/portfolio_history references, add weights/turnover) |
