# ResearchSession slice() Method Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `slice()` method to `ResearchSession` to allow creating a new session with a subset of data filtered by time and symbols.

**Architecture:** The `slice()` method will filter the underlying data (an `AggBar` object) and return a new `ResearchSession` instance with the filtered data while preserving existing session settings.

**Tech Stack:** Python, Polars, Pandas (for timestamp conversion).

---

### Task 1: Add slice() method to ResearchSession

**Files:**
- Modify: `src/factorium/research/session.py`

**Step 1: Write the failing test**

Actually, the user wants me to add the method first, then add tests. But I'll follow TDD as per instructions if possible. Wait, the instructions say "Edit ... add the slice() method" then "Edit ... add tests".

I'll write the tests first to verify they fail.

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/research/test_session.py -v -k slice`
Expected: FAIL (AttributeError: 'ResearchSession' object has no attribute 'slice')

**Step 3: Write minimal implementation**

Add the `slice` method to `ResearchSession` in `src/factorium/research/session.py`.

```python
    def slice(
        self,
        start: Optional[Union[int, str]] = None,
        end: Optional[Union[int, str]] = None,
        symbols: Optional[List[str]] = None,
    ) -> "ResearchSession":
        """
        Create new session with subset of data.

        Args:
            start: Start timestamp (ms or ISO string)
            end: End timestamp (ms or ISO string)
            symbols: Symbol list to include

        Returns:
            New ResearchSession with filtered data
        """
        import pandas as pd

        df = self.data.to_polars()

        # Time filters
        if start is not None:
            if isinstance(start, str):
                start = int(pd.Timestamp(start).value // 1_000_000)
            df = df.filter(pl.col("end_time") >= start)

        if end is not None:
            if isinstance(end, str):
                end = int(pd.Timestamp(end).value // 1_000_000)
            df = df.filter(pl.col("end_time") <= end)

        # Symbol filter
        if symbols is not None:
            df = df.filter(pl.col("symbol").is_in(symbols))

        # Create new session
        from ..aggbar import AggBar

        new_aggbar = AggBar(df)

        return ResearchSession(
            new_aggbar,
            default_frequency=self.default_frequency,
            default_initial_capital=self.default_initial_capital,
            default_transaction_cost=self.default_transaction_cost,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/research/test_session.py -v -k slice`
Expected: PASS

**Step 5: Commit**

```bash
git add -A && git commit -m "fix(research): add slice method to ResearchSession"
```
