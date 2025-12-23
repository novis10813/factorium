# Factorium

A quantitative factor analysis library for financial research.

## Features

- **Factor Operations**: Build complex factors using chainable time-series and cross-sectional operations
- **Multiple Bar Types**: Support for TimeBar, TickBar, VolumeBar, and DollarBar sampling methods
- **Panel Data**: AggBar container for multi-symbol data management
- **High Performance**: Numba-accelerated bar aggregation

## Installation

```bash
# Install in development mode
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from factorium import Factor, AggBar, TimeBar

# Create bars from tick data
bar1 = TimeBar(df1, interval_ms=60_000)  # 1-minute bars
bar2 = TimeBar(df2, interval_ms=60_000)

# Aggregate multiple symbols
agg = AggBar([bar1, bar2])

# Extract factors
close = agg['close']
volume = agg['volume']

# Build alpha factors
momentum = close.ts_delta(20) / close.ts_shift(20)
volatility = close.ts_std(20) / close.ts_mean(20)

# Cross-sectional operations
ranked_momentum = momentum.rank()

# Combine factors
alpha = ranked_momentum * volatility.inverse()
```

## Factor Operations

### Time-Series Operations

| Operation | Description |
|-----------|-------------|
| `ts_rank(window)` | Rolling rank percentile |
| `ts_sum(window)` | Rolling sum |
| `ts_mean(window)` | Rolling mean |
| `ts_std(window)` | Rolling standard deviation |
| `ts_min(window)` | Rolling minimum |
| `ts_max(window)` | Rolling maximum |
| `ts_delta(period)` | Difference from N periods ago |
| `ts_shift(period)` | Shift by N periods |
| `ts_zscore(window)` | Rolling z-score normalization |
| `ts_scale(window)` | Rolling min-max scaling |
| `ts_skewness(window)` | Rolling skewness |
| `ts_kurtosis(window)` | Rolling kurtosis |

### Cross-Sectional Operations

| Operation | Description |
|-----------|-------------|
| `rank()` | Cross-sectional rank percentile |
| `mean()` | Cross-sectional mean |
| `median()` | Cross-sectional median |

### Mathematical Operations

| Operation | Description |
|-----------|-------------|
| `abs()` | Absolute value |
| `sign()` | Sign (-1, 0, 1) |
| `log(base)` | Logarithm |
| `sqrt()` | Square root |
| `pow(exp)` | Power |
| `inverse()` | 1/x |
| `where(cond, other)` | Conditional selection |

### Arithmetic Operations

Factors support standard arithmetic: `+`, `-`, `*`, `/` with other factors or scalars.

## Bar Types

- **TimeBar**: Fixed time intervals (e.g., 1-minute, 1-hour)
- **TickBar**: Fixed number of trades per bar
- **VolumeBar**: Fixed volume per bar
- **DollarBar**: Fixed dollar volume per bar

## License

MIT
