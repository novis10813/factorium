# Migration Guide: v1.x to v2.0

## Breaking Changes

### 1. Backtester Default Changed

```python
# Before: Old iterative implementation
from factorium.backtest import Backtester

# After: Now VectorizedBacktester (Polars-based)
from factorium.backtest import Backtester  # Same import, new implementation

# For old behavior:
from factorium.backtest import LegacyBacktester
```

### 2. BacktestResult Returns Polars

```python
result = bt.run()
result.equity_curve  # Now pl.DataFrame, was pd.Series

# For pandas:
pandas_result = result.to_pandas()
```

### 3. analyze() Returns Dataclass

```python
result = analyzer.analyze()  # Now FactorAnalysisResult
ic_mean = result.ic_summary["mean_ic"]

# For dict:
result_dict = result.to_dict()
```

### 4. Factor.eval() API Changes

**Breaking Changes:**

- Return type changed from `Dict[str, Any]` to `FactorAnalysisResult`
- Parameter `save_path` renamed to `output_dir`
- Parameter `periods` changed from `List[int]` to `int` (MVP, single period only)

**Migration:**

```python
# Before (v0.2.x)
result = factor.eval(
    prices=close,
    periods=[1, 5, 10],
    quantiles=5,
    save_path="./report.png"
)
ic_mean = result["ic_mean"]  # dict access
turnover = result["turnover_mean"]

# After (v0.3.0+)
result = factor.eval(
    prices=close,  # or AggBar
    periods=1,     # single int only (MVP)
    quantiles=5,
    output_dir="./experiments"  # creates timestamped folder
)
ic_mean = result.ic_summary["mean_ic"]  # FactorAnalysisResult access
turnover = result.turnover_mean

# For backward compatibility (dict format):
result_dict = result.to_dict()
```

**New Features:**

- `FactorAnalysisResult.save(output_dir)` method for experiment tracking
- Support for `AggBar` as `prices` parameter (with `price_col` option)
- Turnover metrics included in result (`turnover_series`, `turnover_mean`)

## New Features

### Constraints with normalize
```python
constraint = MaxPositionConstraint(max_weight=0.1, normalize=True)
```

### ResearchSession
```python
from factorium import ResearchSession
session = ResearchSession.load("data.parquet")
signal = session.create_factor("ts_mean(close, 20)", "momentum")
print(session.quick_report(signal))
```

### CompositeFactor
```python
from factorium.factors import CompositeFactor
composite = CompositeFactor.from_zscore([f1, f2])
```
