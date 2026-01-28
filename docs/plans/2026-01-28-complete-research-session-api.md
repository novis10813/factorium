# Complete ResearchSession API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete ResearchSession API by adding analysis integration and data loading shortcuts.

**Architecture:** 
- Update `FactorAnalyzer` to support a unified `analyze()` method and configurable `quantiles`.
- Update `ResearchSession` with `analyze()`, `from_df()`, and `load()` methods.
- Export `ResearchSession` from the main `factorium` package.

**Tech Stack:** Python, Polars, Pandas, pytest

---

### Task 1: Update FactorAnalyzer

**Files:**
- Modify: `src/factorium/factors/analyzer.py`

**Step 1: Update __init__ and add analyze() method**

```python
<<<<
    def __init__(self, factor: Factor, prices: Union[AggBar, Factor]):
        self.factor = factor
        self._raw_prices = prices
====
    def __init__(self, factor: Factor, prices: Union[AggBar, Factor], quantiles: int = 5):
        self.factor = factor
        self.quantiles = quantiles
        self._raw_prices = prices
>>>>
```

Add `analyze()` method:
```python
    def analyze(self) -> dict:
        """
        Perform a standard analysis on the factor.
        
        Returns:
            dict: Analysis results containing IC summary and quantile returns.
        """
        self.prepare_data()
        ic_summary = self.calculate_ic_summary()
        quantile_returns = self.calculate_quantile_returns(quantiles=self.quantiles)
        return {
            "ic_summary": ic_summary,
            "quantile_returns": quantile_returns,
        }
```

**Step 2: Commit**

```bash
git add src/factorium/factors/analyzer.py
git commit -m "feat(analyzer): add analyze() method and quantiles to FactorAnalyzer"
```

---

### Task 2: Update ResearchSession

**Files:**
- Modify: `src/factorium/research/session.py`

**Step 1: Add analyze(), from_df(), and load() methods**

**Step 2: Commit**

```bash
git add src/factorium/research/session.py
git commit -m "feat(research): add analyze(), from_df(), and load() to ResearchSession"
```

---

### Task 3: Update Main Exports

**Files:**
- Modify: `src/factorium/__init__.py`

**Step 1: Add ResearchSession to __all__ and import it**

**Step 2: Commit**

```bash
git add src/factorium/__init__.py
git commit -m "feat: export ResearchSession from main module"
```

---

### Task 4: Add Tests

**Files:**
- Modify: `tests/research/test_session.py`

**Step 1: Add TestResearchSessionMethods class with new tests**

**Step 2: Add TestFactoriumExports class**

**Step 3: Run tests**

Run: `uv run pytest tests/research/ -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add tests/research/test_session.py
git commit -m "test(research): add tests for new ResearchSession methods and exports"
```
