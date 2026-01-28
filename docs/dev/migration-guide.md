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
