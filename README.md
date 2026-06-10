# Factorium

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/factorium.svg)](https://pypi.org/project/factorium/)

Factorium is a **Polars-first factor research and backtesting toolkit** for quantitative finance.

It is designed for researchers who want a notebook-friendly workflow for loading market data, building cross-sectional or time-series factors, evaluating factor quality, and running fast vectorized backtests.

For a Chinese introduction, see [`README_zh.md`](README_zh.md).

## What It Provides

- Data pipeline: `BinanceDataLoader` + `AggBar` for multi-symbol OHLCV panels.
- Factor engine: `Factor` with time-series, cross-sectional, math, and expression operations.
- Analysis: `FactorAnalyzer` and `FactorAnalysisResult` for IC, quantile returns, and plots.
- Backtesting: `VectorizedBacktester`, exposed as `factorium.backtest.Backtester`, with Polars-vectorized PnL.
- Research workflow: `ResearchSession` and `FactorReport` for end-to-end notebook experiments.

## Installation

```bash
uv add factorium
```

Or with pip:

```bash
pip install factorium
```

Development setup:

```bash
git clone https://github.com/novis10813/factorium.git
cd factorium
uv sync --dev
```

## Quick Example

```python
from factorium import ResearchSession

session = ResearchSession.from_parquet("data/btc_1h.parquet")

close = session.factor("close")
momentum = (close.ts_delta(20) / close.ts_shift(20)).cs_rank()

print(session.quick_report(momentum, periods=1))
```

## Documentation

More complete guides live under `docs/`:

- `docs/getting-started/quickstart.md`
- `docs/user-guide/bar.md`
- `docs/user-guide/factor.md`
- `docs/user-guide/analyzer.md`
- `docs/user-guide/backtest.md`

## Status

Factorium is in active alpha development. APIs may still change, but the project already has packaged installation, tests, docs, and examples.

## License

MIT. See `LICENSE`.
