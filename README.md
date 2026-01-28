# Factorium

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/factorium.svg)](https://pypi.org/project/factorium/)

Factorium is a **Polars-first factor research & backtesting toolkit**.

- Data pipeline: `BinanceDataLoader` + `AggBar` → multi-symbol OHLCV panel in one line.
- Factor engine: `Factor` with rich TS/CS/math ops and expression parsing.
- Analysis: `FactorAnalyzer` + `FactorAnalysisResult` for IC, quantile returns, and plots.
- Backtest: `VectorizedBacktester` (exposed as `factorium.backtest.Backtester`) with Polars-vectorized PnL.
- Research workflow: `ResearchSession` + `FactorReport` to tie everything into a notebook-friendly API.

For a Chinese introduction, see `README_zh.md`.

---

## Installation

```bash
# Recommended
uv add factorium

# Or with pip
pip install factorium
```

Development setup:

```bash
git clone https://github.com/novis10813/factorium.git
cd factorium
uv sync --dev
```

---

## Quick example

```python
from factorium import ResearchSession

session = ResearchSession.from_parquet("data/btc_1h.parquet")

close = session.factor("close")
momentum = (close.ts_delta(20) / close.ts_shift(20)).cs_rank()

print(session.quick_report(momentum, periods=1))
```

More complete guides live under `docs/`:

- `docs/getting-started/quickstart.md`
- `docs/user-guide/bar.md`
- `docs/user-guide/factor.md`
- `docs/user-guide/analyzer.md`
- `docs/user-guide/backtest.md`

---

## License

MIT – see `LICENSE`.

