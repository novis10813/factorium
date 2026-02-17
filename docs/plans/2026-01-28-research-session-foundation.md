# ResearchSession Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the `ResearchSession` high-level API for simplified factor research workflows.

**Architecture:** `ResearchSession` acts as a facade over `AggBar`, `Factor`, and `VectorizedBacktester`, providing a more intuitive and fluent API for researchers. It handles data loading (CSV, Parquet) and simplifies backtest configuration by using session-wide defaults.

**Tech Stack:** Python, Polars, Pandas, Pytest, `factorium` (internal package).

---

### Task 1: Initialize Research Module

**Files:**
- Create: `src/factorium/research/__init__.py`
- Create: `tests/research/__init__.py`

**Step 1: Create files**
Create the necessary `__init__.py` files to make the `research` directory a package and to set up the test directory.

**Step 2: Commit**
```bash
git add src/factorium/research/__init__.py tests/research/__init__.py
git commit -m "chore(research): initialize research module"
```

---

### Task 2: Implement ResearchSession Tests

**Files:**
- Create: `tests/research/test_session.py`

**Step 1: Write failing tests**
Copy the test code provided in the task description to `tests/research/test_session.py`.

**Step 2: Run tests to verify they fail**
Run: `uv run pytest tests/research/test_session.py -v`
Expected: FAIL (ModuleNotFoundError or AttributeError because `ResearchSession` isn't implemented yet).

---

### Task 3: Implement ResearchSession Core

**Files:**
- Create: `src/factorium/research/session.py`
- Modify: `src/factorium/research/__init__.py`

**Step 1: Write minimal implementation**
Implement `ResearchSession` in `src/factorium/research/session.py` with the provided code. Update `src/factorium/research/__init__.py` to export `ResearchSession`.

**Step 2: Run tests to verify they pass**
Run: `uv run pytest tests/research/test_session.py -v`
Expected: PASS

**Step 3: Commit**
```bash
git add src/factorium/research/session.py src/factorium/research/__init__.py tests/research/test_session.py
git commit -m "feat(research): add ResearchSession high-level API"
```
