# Factorium Examples

Interactive Jupyter notebooks demonstrating factor research workflows with Factorium.

## Notebooks

| Notebook | Description | Key Concepts |
|----------|-------------|-------------|
| [01 — Momentum Factor Research](01_momentum_factor_research.ipynb) | Complete workflow from data loading to backtest | Data loading, factor construction (code & expression), IC analysis, multi-horizon IC decay, quantile returns, backtesting, quick report |
| [02 — Mean Reversion Factor](02_mean_reversion_factor.ipynb) | Mean reversion with cross-sectional processing | Z-score distance, volatility normalization, `cs_rank`, `cs_zscore`, `cs_winsorize`, market-neutral vs. long-only backtest, advanced operators (`ts_autocorr`, `ts_kurtosis`, `ts_skewness`) |
| [03 — Data Loading & Exploration](03_data_loading_and_exploration.ipynb) | Deep dive into data handling | `BinanceDataLoader`, `AggBar` methods, time-bar intervals (1min/5min/1h), slicing, CSV/Parquet export, `ResearchSession` from files |
| [04 — Multi-Factor Combination](04_multi_factor_combination.ipynb) | Combine and select factors | Factor correlations, `ts_corr`, `cs_neutralize`, `CompositeFactor` (equal/custom/z-score), single vs. composite backtest, factor selection workflow |
| [05 — Universe & Checklist Workflow](05_universe_checklist_workflow.ipynb) | Constrain research and backtest to a tradable asset universe | `Universe`, `Checklist`, `AggBar.with_mask`, `Factor.eval(..., mask=...)`, `Backtester(..., mask=...)` |

## Getting Started

### Prerequisites

```bash
pip install factorium
# or
uv add factorium
```

A Jupyter environment is required:

```bash
pip install jupyterlab
```

### Running the Notebooks

```bash
cd examples/
jupyter lab
```

> **Note:** Notebooks download data directly from Binance Vision. The first run may take a few minutes depending on your internet connection. Subsequent runs are faster thanks to local caching.

## Recommended Reading Order

If you're new to Factorium, we recommend starting with:

1. **Notebook 03** — Understand data loading and the `AggBar` container
2. **Notebook 01** — Walk through a full factor research workflow
3. **Notebook 02** — Learn about signal processing and cross-sectional transforms
4. **Notebook 04** — Combine multiple factors into a composite signal
5. **Notebook 05** — Apply universe/checklist masks consistently in analysis and backtests
