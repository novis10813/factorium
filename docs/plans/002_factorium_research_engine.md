# Plan: Factorium Research Engine (Pre-Evolution Phases)

**Status**: Draft  
**Created**: 2026-01-27  
**Based on**: `001_polars_evolution_engine.md` + Factorium repo

---

## 1. Scope & Non-Goals

This plan defines what the `factorium` package itself should own **before** the evolutionary / storage layers come in.
It is intentionally scoped to **Phase 1–3** of the Polars evolution vision, but **excludes** long-term infrastructure:

- **In scope (Factorium repo)**:
  - Polars-first factor computation engine (TS/CS/Math operators, `Factor` abstraction).
  - High-level research workflow: from raw data → Bars/AggBar → Factors → Analysis → Backtest.
  - Notebook-friendly APIs and plotting utilities.
  - Performance, safety (`safe_*` semantics), and testing strategy.
- **Out of scope (future/other services)**:
  - Factor storage / Knowledge Graph (Polars AST persistence, graph DB).
  - Evolutionary search loop (`EvoEngine`, population management, fitness orchestration).
  - Long-term experiment registry / dashboard / web UI.

The end state of this plan should make Factorium a **standalone, pleasant, and fast factor research lab**, which later
engines (evolutionary search, knowledge graph) can treat as a core dependency.

---

## 2. Vision: A Fast, Polars-Native Research Loop

Design Factorium as a **research loop product**, not just a collection of utilities.

Target user flow:

1. **Load data**
   - From Binance (built-in adapters) or generic CSV/Parquet.
   - Get `Bar` / `AggBar` objects in 1–2 lines of code.
2. **Build factors**
   - Use `Factor` and the expression parser to define TS/CS/Math-based alphas.
   - Support both **code** (`close.ts_mean(20)`) and **string expressions** (`"ts_mean(close, 20)"`).
3. **Analyze**
   - IC / IC decay / quantile returns / turnover / exposure.
   - Quick visualization in notebooks.
4. **Backtest**
   - Cross-sectional long-only / long-short portfolio simulation.
   - Basic transaction cost, slippage and risk constraints.
5. **Iterate**
   - Small diff in factor definition → re-run analysis/backtest quickly.

The guiding principle: **optimize research iteration speed**, leveraging Polars where possible.

---

## 3. Architecture Overview

### 3.1 Data & Container Layer

**Goals**:
- Provide a clean separation between raw market data and factor computations.
- Support multiple bar types and multiple symbols efficiently.

**Key components**:

- `Bar` implementations (`TimeBar`, `TickBar`, `VolumeBar`, `DollarBar`)
  - Input: trade/aggTrade/kline data (pandas/Polars DataFrame).
  - Output: OHLCV bars, optionally with extra engineered features via `.apply`.
- `AggBar`
  - Multi-symbol bar container.
  - Long-format panel (`symbol`, `start_time`, `end_time`, features...).
  - Bridges data layer and factor layer.
- Data loaders
  - `BinanceDataLoader`: opinionated adapter for Binance Vision data (already exists).
  - Future: light-weight generic loader for CSV/Parquet (non-Binance).

Implementation direction:

- Internally prefer **Polars `LazyFrame`** as the main representation where feasible, with conversion hooks to pandas
  for users who need it.
- Maintain `AggBar` API surface but increasingly back it with Polars pipelines.

### 3.2 Factor Engine (Polars-Centric)

**Goals**:
- Make `Factor` the canonical abstraction for “multi-symbol time series factor”, powered by a Polars-first backend.
- Keep operations composable, safe, and introspectable.

**Key ideas**:

- **Backend model**:
  - Short term: `Factor` can wrap either:
    - Eager data (pandas/Polars frame or array).
    - Or a Polars expression + context (aligning with `001_polars_evolution_engine.md`).
  - Medium term: converge on **expression + context** as the primary representation.
- **Operators**:
  - TS operators: `ts_mean`, `ts_std`, `ts_rank`, `ts_corr`, `ts_vr`, etc.
  - CS operators: `rank`, `mean`, etc.
  - Math operators: `abs`, `log`, `signed_pow`, etc.
  - All operators should have a **Polars-native path** with correct `safe_*` semantics.
- **Expression parser**:
  - Continue supporting Alpha101-style string expressions (already exists).
  - Ensure the parser builds an AST that can compile to Polars expressions in the future.

### 3.3 Research & Analysis Layer

**Goals**:
- Provide batteries-included tools to evaluate factors **without** writing a full backtest every time.
- Standardize the metrics and plots Factorium users expect by default.

**Responsibilities**:

- Factor diagnostics:
  - Distribution, coverage (NaNs), outlier statistics.
  - Correlation with basic market variables (return, volatility, volume).
- Cross-sectional performance:
  - IC / rank-IC over time.
  - IC decay across different holding horizons.
  - Quantile portfolio returns (e.g. Q1–Q5).
  - Turnover metrics (per quantile, per factor).
- Visualization:
  - Simple `.plot_*` methods on `Factor` or `Analyzer` objects that return matplotlib figures.

### 3.4 Portfolio Backtest Layer

**Goals**:
- Support factor-driven cross-sectional portfolios with minimal boilerplate.
- Keep the engine **simple but opinionated** rather than a general trading simulator.

**Responsibilities**:

- Portfolio construction:
  - Rank-based long-only (top X%) and long-short (top vs bottom buckets).
  - Multi-factor scoring (weighted sum, z-score combo).
- Constraints:
  - Market/sector neutral toggles (where data allows).
  - Per-symbol weight caps, gross and net exposure caps.
- Mechanics:
  - Rebalance frequency (every bar, daily, weekly).
  - Transaction costs and simple slippage models.
- Outputs:
  - Time series of portfolio returns, drawdown, exposure.
  - Summary metrics (Sharpe, Sortino, turnover, hit rate).

---

## 4. High-Level Roadmap (Within Factorium)

The following phases refer to work happening **inside this repo only**, aligned but not identical to the global Polars
Evolution Engine phases.

### Phase A: Polars Core – Specification & Hardening

This phase assumes that the bulk of the operator migration to Polars has already been completed (TS/CS/Math now use
Polars under the hood). The focus is on **making that work explicit, testable, and future-proof**, not on a large
refactor.

- [ ] Formalize `safe_*` semantics:
  - [ ] Document the exact behavior for NaN propagation, division safety, and `POSITION_EPSILON` for both pandas and Polars.
  - [ ] Add regression tests that lock in these semantics across key operators (e.g. `safe_div`, `ts_mean`, `ts_corr`).
- [ ] Clarify `Factor` backend model:
  - [ ] Write down the intended backends (eager vs expression-backed) and when each is used.
  - [ ] Add tests that validate identical results across backends where both apply.
- [ ] Benchmark and edge-case validation:
  - [ ] Systematize existing benchmarks for large multi-symbol datasets (hook into `2026-01-26-polars-benchmark-design.md`).
  - [ ] Add tests for extreme cases (very long windows, dense NaNs, many symbols) to ensure no performance cliffs or silent misalignment.

### Phase B: Research Session API & Notebook UX

- [ ] Introduce a high-level `ResearchSession` (or similar) API:
  - [ ] Wraps data loading (`BinanceDataLoader` / generic loader).
  - [ ] Manages `AggBar` + factor definitions in one object.
  - [ ] Provides shortcuts for:
    - [ ] Building factors from expressions or callables.
    - [ ] Running standard analyses (IC, quantiles, decay).
    - [ ] Spinning up a basic backtest.
- [ ] Provide canonical example notebooks:
  - [ ] Momentum factor research (multi-symbol, IC + backtest).
  - [ ] Mean-reversion factor with volatility normalization.
  - [ ] Comparison of different bar types (time/tick/volume/dollar).
- [ ] Improve plotting layer:
  - [ ] Standardize styling and figure APIs.
  - [ ] Ensure every major analysis has at least one convenience plot.

### Phase C: Deep Factor Analysis & Portfolio Backtest

- [ ] Extend factor analyzer:
  - [ ] IC breakdown by volatility regime / liquidity bucket (if data available).
  - [ ] Factor correlation matrix and clustering.
  - [ ] Factor orthogonalization utilities (e.g. regress on base factors).
- [ ] Strengthen portfolio backtester:
  - [ ] Rank-to-portfolio pipeline (factor → weights over time).
  - [ ] Simple constraint system (market neutral, caps).
  - [ ] Transaction cost and turnover-aware metrics.
- [ ] Reporting:
  - [ ] Create a minimal report object (or `to_report()` function) that aggregates:
    - [ ] Key charts (equity curve, drawdown, IC time series, quantile returns).
    - [ ] Key statistics tables.

---

## 5. Interfaces & Integration Points

Even though Factorium does **not** own factor storage or evolutionary search, we should design with future integration
in mind.

### 5.1 Factor Introspection

Expose enough metadata for external systems to understand a factor:

- Name / ID.
- Underlying columns used (e.g. `close`, `volume`).
- Operator graph or AST (even if it is not yet fully Polars-native).
- Window sizes, lags, and key hyperparameters.

This metadata can later be serialized by the Knowledge Graph service without changing Factorium’s public API.

### 5.2 Deterministic Evaluation

Ensure that, given:

- Data range (start, end),
- Symbols universe,
- Factor definition,

Factorium computes **deterministic outputs** (up to floating point noise). This is essential so that an external
evolution engine can:

- Treat Factorium as a stable fitness oracle.
- Cache results safely when coupled with data versioning.

### 5.3 Lightweight Serialization Hooks

While Factorium does not implement full storage, it should:

- Provide simple helpers to:
  - Dump factor definitions (expression strings / AST JSON).
  - Save factor panels to Parquet via `AggBar`/`Factor`.
- Avoid committing to a specific DB or graph technology.

---

## 6. Open Questions

- How aggressively should we push towards a **pure Polars expression** backend vs. supporting mixed pandas/Polars flows?
- What is the minimal AST representation that:
  - Feels natural for manual factor writing, and
  - Is still convenient for future evolutionary operators (mutate/crossover)?
- Where is the boundary between:
  - “Rich analysis utilities” (owned by Factorium), and
  - “Experiment management / dashboards” (owned by higher-level tooling)?

These questions do not need to be fully resolved before executing Phases A–C, but should be revisited as the research
engine stabilizes.

