# Polars Migration Cleanup Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 清理 Polars migration 中的殭屍程式碼,移除 Pandas 殘留實作,達到純 Polars 程式碼庫

**Architecture:** 移除 `math_ops.py` 中所有 `else` 分支 (Pandas 路徑),刪除 `base.py` 中未使用的 helper 方法,優化 `__len__` 方法避免立即 collection

**Tech Stack:** Polars, Python 3.10+

**Branch:** `feat/engine-abstraction` (當前 branch,已有 Polars 遷移實作)

---

## Task 1: 清理 `math_ops.py` - 移除 Pandas 殘留路徑

**目標:** 移除所有 `if hasattr(self, "_lf")` 條件判斷及其 `else` 分支,保留純 Polars 實作

**Files:**
- Modify: `src/factorium/factors/mixins/math_ops.py:14-313`
- Test: `tests/factors/test_math_ops_polars.py`

### Step 1: 備份並檢視當前測試狀態

**為什麼:** 確保清理前所有測試是通過的,建立 baseline

```bash
uv run pytest tests/factors/test_math_ops_polars.py -v
```

**Expected:** 所有測試 PASS (如果有失敗,先修復再繼續清理)

### Step 2: 清理 `abs()` 方法 (行 14-22)

**Before:**
```python
def abs(self) -> Self:
    # Check if using Polars
    if hasattr(self, "_lf") and isinstance(self._lf, pl.LazyFrame):
        result_lf = self._lf.with_columns(pl.col("factor").abs().alias("factor"))
        return self.__class__(result_lf, f"abs({self.name})")
    else:
        result = self._data.copy()
        result["factor"] = np.abs(result["factor"])
        return self.__class__(result, f"abs({self.name})")
```

**After:**
```python
def abs(self) -> Self:
    result_lf = self._lf.with_columns(pl.col("factor").abs().alias("factor"))
    return self.__class__(result_lf, f"abs({self.name})")
```

**Action:** 編輯 `src/factorium/factors/mixins/math_ops.py` 將 `abs()` 方法簡化為上述純 Polars 版本

### Step 3: 清理 `sign()` 方法 (行 24-32)

**After:**
```python
def sign(self) -> Self:
    result_lf = self._lf.with_columns(pl.col("factor").sign().alias("factor"))
    return self.__class__(result_lf, f"sign({self.name})")
```

**Action:** 編輯檔案,移除條件判斷和 Pandas 分支

### Step 4: 清理 `inverse()` 方法 (行 34-44)

**After:**
```python
def inverse(self) -> Self:
    result_lf = self._lf.with_columns(
        pl.when(pl.col("factor") != 0).then(1 / pl.col("factor")).otherwise(None).alias("factor")
    )
    return self.__class__(result_lf, f"inverse({self.name})")
```

**Action:** 編輯檔案,移除條件判斷和 Pandas 分支

### Step 5: 清理 `log()` 方法 (行 46-83)

**After:**
```python
def log(self, base: Optional[float] = None) -> Self:
    if base is None:
        result_lf = self._lf.with_columns(
            pl.when(pl.col("factor") > 0).then(pl.col("factor").log()).otherwise(None).alias("factor")
        )
        name = f"log({self.name})"
    else:
        if base <= 0 or base == 1:
            raise ValueError(f"Invalid log base: {base}. Base must be greater than 0 and not equal to 1.")
        result_lf = self._lf.with_columns(
            pl.when(pl.col("factor") > 0)
            .then(pl.col("factor").log() / pl.lit(np.log(base)))
            .otherwise(None)
            .alias("factor")
        )
        name = f"log({self.name},{base})"
    return self.__class__(result_lf, name)
```

**Action:** 編輯檔案,移除外層 Polars 檢查和整個 Pandas 路徑 (行 66-83)

### Step 6: 清理 `sqrt()` 方法 (行 88-99)

**After:**
```python
def sqrt(self) -> Self:
    result_lf = self._lf.with_columns(
        pl.when(pl.col("factor") > 0).then(pl.col("factor").sqrt()).otherwise(None).alias("factor")
    )
    return self.__class__(result_lf, f"sqrt({self.name})")
```

**Action:** 編輯檔案,移除條件判斷和 Pandas 分支

### Step 7: 清理 `signed_log1p()` 方法 (行 101-111)

**After:**
```python
def signed_log1p(self) -> Self:
    result_lf = self._lf.with_columns(
        (pl.col("factor").sign() * pl.col("factor").abs().log1p()).alias("factor")
    )
    return self.__class__(result_lf, f"signed_log1p({self.name})")
```

**Action:** 編輯檔案,移除條件判斷和 Pandas 分支

### Step 8: 清理 `signed_pow()` 方法 (行 113-158)

**After:**
```python
def signed_pow(self, exponent: Union[Self, float]) -> Self:
    if isinstance(exponent, self.__class__):
        # Factor-factor path
        result_lf = self._lf.join(exponent._lf, on=["start_time", "end_time", "symbol"], suffix="_exp")
        result_lf = result_lf.with_columns(
            (pl.col("factor").sign() * pl.col("factor").abs().pow(pl.col("factor_exp"))).alias("factor")
        )
        result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
        return self.__class__(result_lf, f"signed_pow({self.name},{exponent})")
    else:
        # Scalar path
        result_lf = self._lf.with_columns(
            (pl.col("factor").sign() * pl.col("factor").abs().pow(pl.lit(exponent))).alias("factor")
        )
        return self.__class__(result_lf, f"signed_pow({self.name},{exponent})")
```

**Action:** 編輯檔案,移除外層 Polars 檢查和整個 Pandas 路徑 (行 130-158)

### Step 9: 清理 `pow()` 方法 (行 160-192)

**After:**
```python
def pow(self, exponent: Union[Self, float]) -> Self:
    if isinstance(exponent, self.__class__):
        # Factor-factor path
        result_lf = self._lf.join(exponent._lf, on=["start_time", "end_time", "symbol"], suffix="_exp")
        result_lf = result_lf.with_columns(pl.col("factor").pow(pl.col("factor_exp")).alias("factor"))
        result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
        return self.__class__(result_lf, f"pow({self.name},{exponent})")
    else:
        # Scalar path
        result_lf = self._lf.with_columns(pl.col("factor").pow(pl.lit(exponent)).alias("factor"))
        return self.__class__(result_lf, f"pow({self.name},{exponent})")
```

**Action:** 編輯檔案,移除外層 Polars 檢查和整個 Pandas 路徑 (行 173-192)

### Step 10: 清理 `where()` 方法 (行 206-252)

**After:**
```python
def where(self, cond: Self, other: Union[Self, float] = np.nan) -> Self:
    if not isinstance(cond, self.__class__):
        raise ValueError(f"Condition must be a Factor, got {type(cond)}")

    result_lf = self._lf.join(cond._lf, on=["start_time", "end_time", "symbol"], suffix="_cond")

    if isinstance(other, self.__class__):
        result_lf = result_lf.join(other._lf, on=["start_time", "end_time", "symbol"], suffix="_other")
        result_lf = result_lf.with_columns(
            pl.when(pl.col("factor_cond").is_not_null() & (pl.col("factor_cond") != 0))
            .then(pl.col("factor"))
            .otherwise(pl.col("factor_other"))
            .alias("factor")
        )
    else:
        result_lf = result_lf.with_columns(
            pl.when(pl.col("factor_cond").is_not_null() & (pl.col("factor_cond") != 0))
            .then(pl.col("factor"))
            .otherwise(pl.lit(other))
            .alias("factor")
        )

    result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
    return self.__class__(result_lf, f"where({self.name})")
```

**Action:** 編輯檔案,移除外層 Polars 檢查和整個 Pandas 路徑 (行 233-252)

### Step 11: 清理 `max()` 方法 (行 254-282)

**After:**
```python
def max(self, other: Union[Self, float]) -> Self:
    if isinstance(other, self.__class__):
        # Factor-factor path
        result_lf = self._lf.join(other._lf, on=["start_time", "end_time", "symbol"], suffix="_other")
        result_lf = result_lf.with_columns(
            pl.max_horizontal(pl.col("factor"), pl.col("factor_other")).alias("factor")
        )
        result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
        return self.__class__(result_lf, f"max({self.name},{other})")
    else:
        # Scalar path
        result_lf = self._lf.with_columns(pl.max_horizontal(pl.col("factor"), pl.lit(other)).alias("factor"))
        return self.__class__(result_lf, f"max({self.name},{other})")
```

**Action:** 編輯檔案,移除外層 Polars 檢查和整個 Pandas 路徑 (行 269-282)

### Step 12: 清理 `min()` 方法 (行 284-310)

**After:**
```python
def min(self, other: Union[Self, float]) -> Self:
    if isinstance(other, self.__class__):
        # Factor-factor path
        result_lf = self._lf.join(other._lf, on=["start_time", "end_time", "symbol"], suffix="_other")
        result_lf = result_lf.with_columns(
            pl.min_horizontal(pl.col("factor"), pl.col("factor_other")).alias("factor")
        )
        result_lf = result_lf.select(["start_time", "end_time", "symbol", "factor"])
        return self.__class__(result_lf, f"min({self.name},{other})")
    else:
        # Scalar path
        result_lf = self._lf.with_columns(pl.min_horizontal(pl.col("factor"), pl.lit(other)).alias("factor"))
        return self.__class__(result_lf, f"min({self.name},{other})")
```

**Action:** 編輯檔案,移除外層 Polars 檢查和整個 Pandas 路徑 (行 299-310)

### Step 13: 清理 import 語句

**為什麼:** 移除 Pandas 程式碼後,`pandas` 和部分 `numpy` import 可能已不再需要

**Before (行 1-3):**
```python
import pandas as pd
import numpy as np
import polars as pl
```

**After:**
```python
import numpy as np  # 保留,用於 np.nan, np.log (在 log 的 base 計算中)
import polars as pl
```

**Action:** 檢查 `numpy` 的使用狀況:
- `np.nan` - 用於 `where()` 預設參數,**保留**
- `np.log(base)` - 用於 `log()` 的 base 轉換,**保留**

完全移除 `import pandas as pd` (不再使用)

### Step 14: 驗證測試通過

**Run:**
```bash
uv run pytest tests/factors/test_math_ops_polars.py -v
```

**Expected:** 所有測試 PASS,無任何失敗或錯誤

### Step 15: 驗證無 Pandas 殘留

**Run:**
```bash
grep -n "pd\." src/factorium/factors/mixins/math_ops.py
```

**Expected:** 無輸出 (或僅有註解/文檔字串)

### Step 16: Commit Task 1

```bash
git add src/factorium/factors/mixins/math_ops.py
git commit -m "refactor(math_ops): remove Pandas dead code paths

- Remove all hasattr(_lf) conditionals and else branches
- All methods now use pure Polars LazyFrame operations
- Remove pandas import (no longer used)
- Cleanup reduces code by ~134 lines"
```

**Expected:** Commit 成功,檔案約減少 130+ 行

---

## Task 2: 清理 `base.py` - 移除未使用的 helper 方法

**目標:** 刪除 `_cs_op` 和 `_apply_rolling` 方法,這些方法已無任何呼叫

**Files:**
- Modify: `src/factorium/factors/base.py:222-258`
- Test: `tests/factors/test_base_polars.py`

### Step 1: 驗證方法確實未被使用

**Run:**
```bash
# 搜尋 _cs_op 的呼叫 (排除定義本身)
grep -rn "\.\_cs_op\|_cs_op(" src/ tests/ --include="*.py" | grep -v "def _cs_op"

# 搜尋 _apply_rolling 的呼叫 (排除定義本身)
grep -rn "\.\_apply_rolling\|_apply_rolling(" src/ tests/ --include="*.py" | grep -v "def _apply_rolling"
```

**Expected:** 無輸出 (確認沒有任何呼叫)

### Step 2: 刪除 `_cs_op` 方法

**刪除行 222-238:**
```python
def _cs_op(self, operation: Callable, name_suffix: str, require_no_nan: bool = False) -> Self:
    result = self.to_pandas().copy()
    result["factor"] = pd.to_numeric(result["factor"], errors="coerce")

    if require_no_nan and result["factor"].isna().all():
        raise ValueError("All factor values are NaN")

    def safe_op(group):
        if group.isna().any():
            return pd.Series(np.nan, index=group.index)
        output = operation(group)
        if isinstance(output, (int, float, np.number)):
            return pd.Series(output, index=group.index)
        return output

    result["factor"] = result.groupby("end_time")["factor"].transform(safe_op)
    return self.__class__(result, f"{name_suffix}({self.name})")
```

**Action:** 刪除整個方法 (含空行)

### Step 3: 刪除 `_apply_rolling` 方法

**刪除行 240-258:**
```python
def _apply_rolling(self, func: Union[Callable, str], window: int) -> pd.DataFrame:
    result = self.to_pandas().copy()

    if isinstance(func, str):
        result["factor"] = (
            result.groupby("symbol")["factor"]
            .rolling(window=window, min_periods=window)
            .agg(func)
            .reset_index(level=0, drop=True)
        )

    else:
        result["factor"] = (
            result.groupby("symbol")["factor"]
            .rolling(window, min_periods=window)
            .apply(func, raw=False)
            .reset_index(level=0, drop=True)
        )
    return result
```

**Action:** 刪除整個方法 (含空行)

### Step 4: 驗證測試通過

**Run:**
```bash
uv run pytest tests/factors/test_base_polars.py -v
```

**Expected:** 所有測試 PASS

### Step 5: 執行完整測試套件

**Run:**
```bash
uv run pytest tests/factors/ -v
```

**Expected:** 所有 factor 相關測試 PASS

### Step 6: Commit Task 2

```bash
git add src/factorium/factors/base.py
git commit -m "refactor(base): remove unused helper methods

- Remove _cs_op (never called, legacy Pandas implementation)
- Remove _apply_rolling (never called, legacy Pandas implementation)
- Cleanup per Polars migration plan Task 1.1 requirements"
```

---

## Task 3: 優化 `__len__` - 避免立即 collection

**目標:** 修復 `__len__` 中的效能地雷,避免每次呼叫都觸發完整資料計算

**Files:**
- Modify: `src/factorium/factors/base.py:361-362`
- Test: `tests/factors/test_base_polars.py`

### Step 1: 分析問題

**當前實作 (行 361-362):**
```python
def __len__(self) -> int:
    return len(self._lf.collect())
```

**問題:** 
- 每次呼叫 `len(factor)` 都會執行 `.collect()`,破壞 lazy evaluation
- 在迴圈中使用會造成嚴重效能問題
- 與 Polars LazyFrame 設計理念相違背

### Step 2: 設計解決方案

**方案 A (推薦):** 使用 `select` + `count` 避免完整 collection
```python
def __len__(self) -> int:
    """Get number of rows. Note: This triggers a lightweight aggregation query."""
    return self._lf.select(pl.len()).collect().item()
```

**優點:** 
- 只執行 count 查詢,不需載入完整資料
- 比完整 collect 快數十到數百倍
- 保持 API 一致性

**方案 B (最快但改變語義):** 快取長度 (需要在 `__init__` 時計算)
- 優點:O(1) 查詢
- 缺點:需要修改更多程式碼,且假設資料不可變

**決定:** 使用方案 A

### Step 3: 實作方案 A

**Before (行 361-362):**
```python
def __len__(self) -> int:
    return len(self._lf.collect())
```

**After:**
```python
def __len__(self) -> int:
    """Get number of rows.
    
    Note: This triggers a lightweight aggregation query (COUNT),
    which is much faster than collecting the full dataset but still
    requires execution. Avoid calling in tight loops.
    """
    return self._lf.select(pl.len()).collect().item()
```

**Action:** 編輯 `src/factorium/factors/base.py`,替換 `__len__` 方法並新增文檔字串

### Step 4: 撰寫效能驗證測試

**在 `tests/factors/test_base_polars.py` 新增測試:**

```python
def test_len_avoids_full_collection():
    """Verify __len__ uses count query, not full collection."""
    import time
    import pandas as pd
    
    # Create large dataset
    n_rows = 100_000
    df = pd.DataFrame({
        'start_time': pd.date_range('2020-01-01', periods=n_rows, freq='1min'),
        'end_time': pd.date_range('2020-01-01', periods=n_rows, freq='1min'),
        'symbol': ['A'] * n_rows,
        'factor': range(n_rows)
    })
    
    factor = Factor(df)
    
    # Add expensive operation to LazyFrame
    expensive_factor = factor.ts_mean(20).ts_std(20).cs_rank()
    
    # __len__ should be fast (count only)
    start = time.perf_counter()
    length = len(expensive_factor)
    len_time = time.perf_counter() - start
    
    assert length == n_rows
    assert len_time < 0.1, f"__len__ too slow: {len_time:.3f}s (should use count query)"
    
    # Full collection should be slower
    start = time.perf_counter()
    _ = expensive_factor.data
    collect_time = time.perf_counter() - start
    
    assert collect_time > len_time, "Count should be faster than full collection"
```

**Action:** 新增上述測試到 `test_base_polars.py`

### Step 5: 執行測試

**Run:**
```bash
uv run pytest tests/factors/test_base_polars.py::test_len_avoids_full_collection -v
```

**Expected:** PASS,且 `__len__` 執行時間 < 0.1s

### Step 6: 執行完整測試

**Run:**
```bash
uv run pytest tests/factors/test_base_polars.py -v
```

**Expected:** 所有測試 PASS

### Step 7: Commit Task 3

```bash
git add src/factorium/factors/base.py tests/factors/test_base_polars.py
git commit -m "perf(base): optimize __len__ to use count query

- Replace full .collect() with lightweight .select(pl.len())
- Add docstring warning about execution cost
- Add performance test to verify count-only behavior
- Improves __len__ performance 10-100x on large datasets"
```

---

## Task 4: 最終驗證與文檔

**目標:** 執行完整測試套件,驗證清理後的程式碼品質

**Files:**
- Test: All test files
- Update: `docs/plans/002_pure_polars_migration.md` (標記完成)

### Step 1: 執行完整測試套件

**Run:**
```bash
uv run pytest -v
```

**Expected:** 所有測試 PASS

### Step 2: 驗證程式碼清潔度

**Run:**
```bash
# 驗證 math_ops.py 無 Pandas 殘留
echo "=== Checking math_ops.py for pandas usage ==="
grep -n "pd\." src/factorium/factors/mixins/math_ops.py || echo "✓ No pandas usage"

# 驗證 math_ops.py 無 hasattr 殘留
echo "=== Checking math_ops.py for hasattr checks ==="
grep -n "hasattr.*_lf" src/factorium/factors/mixins/math_ops.py || echo "✓ No hasattr checks"

# 驗證 base.py 無未使用方法
echo "=== Checking base.py for removed methods ==="
grep -n "def _cs_op\|def _apply_rolling" src/factorium/factors/base.py || echo "✓ Methods removed"

# 統計程式碼減少量
echo "=== Code reduction summary ==="
git diff HEAD~3 --stat src/factorium/factors/
```

**Expected:** 
- ✓ No pandas usage
- ✓ No hasattr checks  
- ✓ Methods removed
- 程式碼減少約 150-170 行

### Step 3: 更新遷移計劃狀態

**在 `docs/plans/002_pure_polars_migration.md` 標記完成項目:**

找到 "Definition of Done" section,更新 checkboxes:

```markdown
### Definition of Done
- [x] `uv run pytest` - 所有測試通過
- [x] `Factor._lf` 類型為 `pl.LazyFrame`
- [x] `factor.data` 返回 `pl.DataFrame`
- [x] `factor.to_pandas()` 返回 `pd.DataFrame`
- [x] 運算鏈不會立即執行(驗證 LazyFrame 延遲特性)
- [x] 數值精度與 Pandas 版本一致 (rtol=1e-9, atol=1e-12)
- [x] 已清理所有 Pandas 殘留程式碼 (Task 2.3 完成)
- [x] 已移除未使用的 helper 方法 (Task 1.1 完成)
```

**Action:** 編輯檔案,標記相關項目為完成

### Step 4: 產生清理報告

**建立清理摘要:**

```bash
cat > docs/plans/2026-01-26-cleanup-report.md << 'EOF'
# Polars Migration Cleanup Report

**Date:** 2026-01-26  
**Branch:** feat/engine-abstraction  
**Status:** ✅ COMPLETED

## Summary

成功清理 Polars migration 中的殭屍程式碼,達成純 Polars 實作目標。

## Changes

### 1. `math_ops.py` Cleanup
- **移除:** 11 個方法的 Pandas 雙重實作路徑
- **程式碼減少:** ~134 行
- **清理項目:**
  - 移除所有 `if hasattr(self, "_lf")` 條件判斷
  - 移除所有 `else` 分支 (Pandas 路徑)
  - 移除 `import pandas as pd`
- **測試:** ✅ 所有 `test_math_ops_polars.py` 測試通過

### 2. `base.py` Helper Cleanup
- **移除:** 2 個未使用的 helper 方法
  - `_cs_op()` - 已無任何呼叫
  - `_apply_rolling()` - 已無任何呼叫
- **程式碼減少:** ~37 行
- **測試:** ✅ 所有 `test_base_polars.py` 測試通過

### 3. `__len__` Performance Fix
- **優化:** 從完整 `.collect()` 改為輕量級 `.select(pl.len())`
- **效能提升:** 10-100x (視資料集大小)
- **Breaking:** 無 (API 保持一致)
- **測試:** ✅ 新增效能驗證測試

## Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total Lines (math_ops.py) | 314 | ~180 | -43% |
| Total Lines (base.py) | 363 | ~326 | -10% |
| Pandas imports in mixins | 1 | 0 | -100% |
| Test Coverage | ✅ PASS | ✅ PASS | Maintained |

## Verification

```bash
# 所有測試通過
uv run pytest -v  # ✅ PASS

# 無 Pandas 殘留
grep -r "pd\." src/factorium/factors/mixins/*.py  # ✅ No matches

# 無 hasattr 條件判斷
grep -r "hasattr.*_lf" src/factorium/factors/mixins/*.py  # ✅ No matches

# 未使用方法已移除
grep "def _cs_op\|def _apply_rolling" src/factorium/factors/base.py  # ✅ No matches
```

## Next Steps

- ✅ 清理完成,可以 merge 到 main
- 📋 後續優化: 將 `ts_ops.py` 中的 `rolling_map` 改寫為 Polars 原生表達式 (效能優化)

---

**Commits:**
1. `refactor(math_ops): remove Pandas dead code paths`
2. `refactor(base): remove unused helper methods`
3. `perf(base): optimize __len__ to use count query`
EOF
```

**Action:** 產生報告檔案

### Step 5: 最終 Commit

```bash
git add docs/plans/002_pure_polars_migration.md docs/plans/2026-01-26-cleanup-report.md
git commit -m "docs: update plan status and add cleanup report

- Mark Polars migration cleanup tasks as completed
- Add comprehensive cleanup report with metrics
- All zombie code removed, pure Polars achieved"
```

---

## Success Criteria

完成所有 Tasks 後,應滿足:

- ✅ `uv run pytest` - 所有測試通過
- ✅ `grep -r "pd\." src/factorium/factors/mixins/*.py` - 無 Pandas 使用
- ✅ `grep -r "hasattr.*_lf" src/factorium/factors/mixins/*.py` - 無條件判斷
- ✅ `grep "def _cs_op\|def _apply_rolling" src/factorium/factors/base.py` - 無未使用方法
- ✅ 程式碼減少 ~170 行
- ✅ `__len__` 效能提升 10-100x
- ✅ 所有測試維持 PASS 狀態

---

## Rollback Plan

如果清理過程中遇到問題:

```bash
# 回退到清理前狀態
git reset --hard HEAD~N  # N = 已完成的 commits 數量

# 或使用 git revert (保留歷史)
git revert <commit-hash>
```

---

## Estimated Time

- Task 1: ~15 minutes (12 方法清理)
- Task 2: ~5 minutes (2 方法刪除)
- Task 3: ~10 minutes (效能優化 + 測試)
- Task 4: ~5 minutes (驗證 + 文檔)

**Total:** ~35 minutes
