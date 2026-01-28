# Fix Task C.4 - FactorReport generate() Method Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `FactorReport.generate()` method to automate factor analysis and backtesting.

**Architecture:** Add a classmethod `generate` to `FactorReport` that takes a `ResearchSession` and `Factor`, calls `session.analyze()` and `session.backtest()`, and returns a `FactorReport` instance.

**Tech Stack:** Python, Polars/Pandas, Pytest.

---

### Task 1: Update FactorReport class

**Files:**
- Modify: `src/factorium/research/report.py`

**Step 1: Add TYPE_CHECKING and imports**
Add `TYPE_CHECKING` to imports and import `ResearchSession` under `TYPE_CHECKING` to avoid circular imports.

**Step 2: Add generate classmethod**
Implement the `generate` method as specified.

```python
    @classmethod
    def generate(
        cls,
        session: "ResearchSession",
        factor: "Factor",
        price_col: str = "close",
        quantiles: int = 5,
        **backtest_kwargs
    ) -> "FactorReport":
        """
        Generate report by running analysis and backtest automatically.
        
        Args:
            session: ResearchSession with data
            factor: Factor to analyze
            price_col: Price column for analysis
            quantiles: Quantiles for analysis
            **backtest_kwargs: Additional args for backtest (neutralization, etc.)
        
        Returns:
            FactorReport with complete analysis and backtest results
        """
        # Run analysis
        analysis = session.analyze(factor, price_col=price_col, quantiles=quantiles)
        
        # Run backtest
        backtest = session.backtest(factor, **backtest_kwargs)
        
        return cls(factor, analysis, backtest)
```

### Task 2: Add test for FactorReport.generate()

**Files:**
- Modify: `tests/research/test_report.py`

**Step 1: Add test_generate_automates_workflow**
Add the test case to `TestFactorReport` class.

### Task 3: Verification

**Step 1: Run tests**
Run: `uv run pytest tests/research/test_report.py -v`
Expected: All tests pass, including the new one.

### Task 4: Commit

**Step 1: Commit changes**
Commit with message: "feat: add FactorReport.generate() to automate analysis and backtesting"
