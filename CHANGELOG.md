# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2026-02-17
### Added
- VectorizedBacktester: standardized `signal -> exposure -> weight` pipeline and portfolio schemes (market‑neutral, long‑only, top‑N patterns). (see #5)
- Factor analysis: multi‑horizon IC decay and flexible targets for `FactorAnalyzer` / reports. (see #4)
- Factor correlation utilities and clustering analysis (correlation matrix + visualizations). (see #6)
- Factor orthogonalization utilities (`cs_neutralize` / residual‑based orthogonalization). (see #7)
- Additional backtest metrics: Sortino ratio, Calmar ratio, win rate and improved metrics handling.
- New example notebooks demonstrating multi‑factor workflows and orthogonalization (`examples/04_multi_factor_combination.ipynb`).
- Extensive unit and integration tests for backtest, factor ops, and Polars paths.

### Changed
- `Backtester` is now an alias for `VectorizedBacktester` (Polars‑based implementation).
- Internal refactors and Polars migration improvements for TS/CS operators and analyzer.

### Fixed
- Various bug fixes and test stabilizations across data loading and backtest path.

### Notes
- Backward compatibility: No breaking API changes expected for typical user workflows. See `docs/dev/migration-guide.md` for migration notes if you rely on internal/edge APIs.

---

(Full changelog & commit list available in the release PR.)
