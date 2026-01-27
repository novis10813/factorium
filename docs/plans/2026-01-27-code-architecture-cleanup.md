# Code Review & Architecture Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 統一 Epsilon 常數、修復 Safe Division 問題、清理 Engine 架構重複。

**Architecture:** 建立 `constants.py` 作為共用常數模組，修復所有 `1e-10` 硬編碼為 `EPSILON`。將舊 `engine/` 目錄移至 `tests/_legacy_engine/` 僅供一致性測試使用。

**Tech Stack:** Python, Polars, pytest

---

## Phase 1: Constants Module & Safe Division

### Task 1: 建立 constants.py 模組

**Files:**
- Create: `src/factorium/constants.py`
- Test: 無需新測試（常數模組）

**Step 1: 建立 constants.py**

```python
"""Constants used across the factorium package."""

# Numerical precision constants
EPSILON = 1e-10  # General numerical epsilon for safe division

# Time constants (reference, used in backtest)
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
MIN_PERIODS_PER_YEAR = 1.0
MAX_PERIODS_PER_YEAR = 365.25 * 24 * 60
```

**Step 2: 更新 backtest/utils.py 使用 constants.py**

修改 `src/factorium/backtest/utils.py`，從 constants.py 導入：

```python
# 移除舊定義，改為 re-export
from ..constants import EPSILON, SECONDS_PER_YEAR, MIN_PERIODS_PER_YEAR, MAX_PERIODS_PER_YEAR

# 保留 POSITION_EPSILON 作為別名（向後相容）
POSITION_EPSILON = EPSILON
```

**Step 3: 執行測試確認沒有 breaking change**

```bash
pytest tests/backtest/ -v --tb=short
```

Expected: All PASS

**Step 4: Commit**

```bash
git add src/factorium/constants.py src/factorium/backtest/utils.py
git commit -m "refactor: extract constants to dedicated module"
```

---

### Task 2: 修復 base.py 的 Safe Division

**Files:**
- Modify: `src/factorium/factors/base.py:261, 272-276, 284`

**Step 1: 更新 base.py import 並替換 1e-10**

在檔案頂部加入：
```python
from ..constants import EPSILON
```

修改 `__truediv__` 方法 (line 261):
```python
# Before:
pl.when(pl.col("other").abs() <= 1e-10)

# After:
pl.when(pl.col("other").abs() <= EPSILON)
```

修改 scalar division (line 272-276):
```python
# Before:
result_lf = self._lf.with_columns((pl.col("factor") / pl.lit(other)).alias("factor"))

# After:
result_lf = self._lf.with_columns(
    pl.when(pl.lit(other).abs() <= EPSILON)
    .then(pl.lit(None))
    .otherwise(pl.col("factor") / pl.lit(other))
    .alias("factor")
)
```

修改 `__rtruediv__` 方法 (line 284):
```python
# Before:
pl.when(pl.col("factor").abs() <= 1e-10)

# After:
pl.when(pl.col("factor").abs() <= EPSILON)
```

**Step 2: 執行測試**

```bash
pytest tests/factors/test_factor.py -v --tb=short
```

Expected: All PASS

**Step 3: Commit**

```bash
git add src/factorium/factors/base.py
git commit -m "fix(base): use EPSILON constant and fix scalar safe division"
```

---

### Task 3: 修復 math_ops.py 的 inverse() 方法

**Files:**
- Modify: `src/factorium/factors/mixins/math_ops.py:21-25`

**Step 1: 更新 math_ops.py**

在檔案頂部加入：
```python
from ...constants import EPSILON
```

修改 `inverse()` 方法:
```python
# Before:
def inverse(self) -> Self:
    result_lf = self._lf.with_columns(
        pl.when(pl.col("factor") != 0).then(1 / pl.col("factor")).otherwise(None).alias("factor")
    )
    return self.__class__(result_lf, f"inverse({self.name})")

# After:
def inverse(self) -> Self:
    result_lf = self._lf.with_columns(
        pl.when(pl.col("factor").abs() <= EPSILON)
        .then(pl.lit(None))
        .otherwise(1 / pl.col("factor"))
        .alias("factor")
    )
    return self.__class__(result_lf, f"inverse({self.name})")
```

**Step 2: 執行測試**

```bash
pytest tests/factors/test_factor.py::TestMathOps -v --tb=short
```

Expected: All PASS

**Step 3: Commit**

```bash
git add src/factorium/factors/mixins/math_ops.py
git commit -m "fix(math_ops): use EPSILON for inverse() safety check"
```

---

### Task 4: 更新 ts_ops.py 的 1e-10 常數

**Files:**
- Modify: `src/factorium/factors/mixins/ts_ops.py` (lines 251, 272, 383, 430, 497, 560, 585, 660)

**Step 1: 加入 import**

```python
from ...constants import EPSILON
```

**Step 2: 全域替換**

使用搜尋替換：`1e-10` → `EPSILON`

**注意**: 僅替換用於除法安全檢查的 `1e-10`。如果有用於數值穩定性的（如 `mean + 1e-10` 防止 log(0)），需保留或使用 `NUMERICAL_STABILITY_EPSILON` 區分。

**Step 3: 執行測試**

```bash
pytest tests/factors/test_factor.py::TestTimeSeriesOps -v --tb=short
```

Expected: All PASS

**Step 4: Commit**

```bash
git add src/factorium/factors/mixins/ts_ops.py
git commit -m "refactor(ts_ops): use EPSILON constant for safe division"
```

---

### Task 5: 更新 cs_ops.py 的 1e-10 常數

**Files:**
- Modify: `src/factorium/factors/mixins/cs_ops.py:94`

**Step 1: 加入 import 並替換**

```python
from ...constants import EPSILON
```

替換 line 94 的 `1e-10` → `EPSILON`

**Step 2: 執行測試**

```bash
pytest tests/factors/test_factor.py::TestCrossSectionOps -v --tb=short
```

Expected: All PASS

**Step 3: Commit**

```bash
git add src/factorium/factors/mixins/cs_ops.py
git commit -m "refactor(cs_ops): use EPSILON constant for safe division"
```

---

### Task 6: 更新 engine.py 的 1e-10 常數

**Files:**
- Modify: `src/factorium/factors/engine.py` (line 235)

**Step 1: 加入 import 並替換**

```python
from ..constants import EPSILON
```

替換所有 `1e-10` → `EPSILON`

**Step 2: 執行測試**

```bash
pytest tests/factors/test_engine.py -v --tb=short
```

Expected: All PASS

**Step 3: Commit**

```bash
git add src/factorium/factors/engine.py
git commit -m "refactor(engine): use EPSILON constant for safe division"
```

---

## Phase 2: Architecture Cleanup

### Task 7: 移動舊 Engine 到測試目錄

**Files:**
- Move: `src/factorium/factors/engine/` → `tests/factors/_legacy_engine/`

**Step 1: 移動目錄**

```bash
mv src/factorium/factors/engine tests/factors/_legacy_engine
```

**Step 2: 更新 _legacy_engine 中的 import 路徑**

修改 `tests/factors/_legacy_engine/__init__.py`:
```python
# 保持不變，但不再對外暴露
from .polars import PolarsEngine
from .pandas import PandasEngine

__all__ = ["PolarsEngine", "PandasEngine"]
```

修改 `tests/factors/_legacy_engine/polars.py` 和 `pandas.py` 的 import:
```python
# 如果有相對導入指向 factorium.factors，改為絕對導入
# 例如: from ..base import ... → from factorium.factors.base import ...
```

**Step 3: 更新 _legacy_engine 中的 1e-10**

```python
from factorium.constants import EPSILON
```

替換所有 `1e-10` → `EPSILON`

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move legacy engine to tests for consistency testing only"
```

---

### Task 8: 更新測試導入路徑

**Files:**
- Modify: `tests/factors/test_engine_consistency.py`

**Step 1: 更新導入**

```python
# Before:
from factorium.factors.engine import PolarsEngine, PandasEngine

# After:
from tests.factors._legacy_engine import PolarsEngine, PandasEngine
# 或使用相對導入
from ._legacy_engine import PolarsEngine, PandasEngine
```

**Step 2: 執行一致性測試**

```bash
pytest tests/factors/test_engine_consistency.py -v --tb=short
```

Expected: All PASS

**Step 3: Commit**

```bash
git add tests/factors/test_engine_consistency.py
git commit -m "test: update legacy engine import path for consistency tests"
```

---

### Task 9: 清理 factors/__init__.py

**Files:**
- Modify: `src/factorium/factors/__init__.py`

**Step 1: 確認只導出新架構**

```python
# 確保只有這些導出（已經是正確的）
from .base import BaseFactor
from .core import Factor
from .engine import PolarsEngine  # 指向 engine.py（新架構）
from .parser import FactorExpressionParser
from .analyzer import FactorAnalyzer

__all__ = [
    "BaseFactor",
    "Factor",
    "PolarsEngine",
    "FactorExpressionParser",
    "FactorAnalyzer",
]
```

**Step 2: 執行完整測試**

```bash
pytest tests/factors/ -v --tb=short
```

Expected: All PASS

**Step 3: Commit**

```bash
git add src/factorium/factors/__init__.py
git commit -m "refactor(factors): ensure clean public API exports"
```

---

## Phase 3: Final Verification

### Task 10: 執行完整測試套件

**Step 1: 執行所有測試**

```bash
pytest -v --tb=short
```

Expected: All 478+ tests PASS

**Step 2: 檢查是否有遺漏的 1e-10**

```bash
rg "1e-10" src/factorium --type py
```

Expected: 無輸出（或僅有特殊用途的數值穩定性常數）

**Step 3: Final Commit (如有需要)**

```bash
git status
# 如有遺漏的變更，補上 commit
```

---

## Summary

| Phase | Tasks | 目標 |
|-------|-------|------|
| 1 | Task 1-6 | Constants 統一 + Safe Division 修復 |
| 2 | Task 7-9 | Architecture 清理 |
| 3 | Task 10 | 最終驗證 |

**預估時間**: 30-45 分鐘

**風險**:
- `_legacy_engine` 中的 import 路徑可能需要調整
- 某些 `1e-10` 可能是數值穩定性用途，需個別判斷

**回滾**:
- 每個 Task 有獨立 commit，可以個別 revert
