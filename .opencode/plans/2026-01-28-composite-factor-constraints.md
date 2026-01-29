# CompositeFactor and WeightConstraint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement multi-factor composition (CompositeFactor) and weight constraints (WeightConstraint, MaxPositionConstraint, LongOnlyConstraint) for the research engine.

**Architecture:** 
- `CompositeFactor` allows combining multiple `Factor` objects using weighted combinations. It joins the factors on time and symbol, and calculates the weighted sum.
- `WeightConstraint` is an abstract base class for modifying position weights. `MaxPositionConstraint` clips weights to a maximum absolute value, and `LongOnlyConstraint` ensures weights are non-negative.

**Tech Stack:** Python, Polars, Pytest

---

### Task 1: Implement CompositeFactor

**Files:**
- Create: `src/factorium/factors/composite.py`
- Modify: `src/factorium/factors/__init__.py`
- Test: `tests/factors/test_composite.py`

**Step 1: Write the failing test**
Create `tests/factors/test_composite.py` with the provided tests.

**Step 2: Run test to verify it fails**
Run: `pytest tests/factors/test_composite.py`
Expected: FAIL (ModuleNotFoundError or ImportError)

**Step 3: Write minimal implementation**
Create `src/factorium/factors/composite.py` and update `src/factorium/factors/__init__.py`.

**Step 4: Run test to verify it passes**
Run: `pytest tests/factors/test_composite.py`
Expected: PASS

**Step 5: Commit**
```bash
git add src/factorium/factors/composite.py src/factorium/factors/__init__.py tests/factors/test_composite.py
git commit -m "feat: add CompositeFactor for multi-factor composition"
```

---

### Task 2: Implement WeightConstraint

**Files:**
- Create: `src/factorium/backtest/constraints.py`
- Modify: `src/factorium/backtest/__init__.py`
- Test: `tests/backtest/test_constraints.py`

**Step 1: Write the failing test**
Create `tests/backtest/test_constraints.py` with the provided tests.

**Step 2: Run test to verify it fails**
Run: `pytest tests/backtest/test_constraints.py`
Expected: FAIL (ModuleNotFoundError or ImportError)

**Step 3: Write minimal implementation**
Create `src/factorium/backtest/constraints.py` and update `src/factorium/backtest/__init__.py`.

**Step 4: Run test to verify it passes**
Run: `pytest tests/backtest/test_constraints.py`
Expected: PASS

**Step 5: Commit**
```bash
git add src/factorium/backtest/constraints.py src/factorium/backtest/__init__.py tests/backtest/test_constraints.py
git commit -m "feat: add WeightConstraint system"
```
