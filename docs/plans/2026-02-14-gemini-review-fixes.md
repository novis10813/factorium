# Gemini Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修正外部 code review 指出的高/中/低優先級問題，確保時區一致性、互動式環境相容性、資料準備狀態一致性與回測權重流程簡化。

**Architecture:** 以最小侵入修補既有流程，優先處理會造成 RuntimeError 或跨時區資料錯位的問題。每個修正都先補測試再改實作，避免回歸。針對「可能很慢」的行為採文檔/參數防呆而非過度重構，維持現有 API 穩定。

**Tech Stack:** Python 3.10+, asyncio, pandas/polars, pytest

---

### Task 1: 修正 date range 以 UTC midnight 對齊

**Files:**
- Modify: `src/factorium/data/utils.py:11`
- Test: `tests/data/test_timestamp_utils.py`

**Step 1: Write the failing test**

```python
def test_calculate_date_range_uses_utc_midnight(monkeypatch):
    # mock 現在時間為非 UTC 本地時區，驗證 daily 邊界仍以 UTC 00:00 對齊
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_timestamp_utils.py::test_calculate_date_range_uses_utc_midnight -v`
Expected: FAIL，顯示回傳邊界不是 UTC midnight。

**Step 3: Write minimal implementation**

```python
from datetime import timezone

today_midnight = datetime.now(timezone.utc).replace(
    hour=0, minute=0, second=0, microsecond=0
)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_timestamp_utils.py::test_calculate_date_range_uses_utc_midnight -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/data/test_timestamp_utils.py src/factorium/data/utils.py
git commit -m "fix(data): align date range boundaries to UTC midnight"
```

### Task 2: 修正 notebook/互動式環境 asyncio 相容性

**Files:**
- Modify: `src/factorium/universe/metadata.py:49`
- Modify: `src/factorium/universe/tags.py:83`
- Reference: `src/factorium/data/loader.py`（`_run_async`）
- Test: `tests/universe/test_metadata.py`
- Test: `tests/universe/test_tags.py`

**Step 1: Write the failing tests**

```python
def test_metadata_fetch_uses_run_async(monkeypatch):
    ...

def test_tags_fetch_uses_run_async(monkeypatch):
    ...
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/universe/test_metadata.py::test_metadata_fetch_uses_run_async tests/universe/test_tags.py::test_tags_fetch_uses_run_async -v`
Expected: FAIL，顯示目前仍呼叫 `asyncio.run`。

**Step 3: Write minimal implementation**

```python
from ..data.loader import _run_async

def fetch(...):
    return _run_async(self.fetch_async(...))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/universe/test_metadata.py::test_metadata_fetch_uses_run_async tests/universe/test_tags.py::test_tags_fetch_uses_run_async -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/universe/test_metadata.py tests/universe/test_tags.py src/factorium/universe/metadata.py src/factorium/universe/tags.py
git commit -m "fix(universe): avoid asyncio.run in sync fetch helpers"
```

### Task 3: 修正 Analyzer prepared data 與 period cache 一致性

**Files:**
- Modify: `src/factorium/factors/analyzer.py:244`
- Test: `tests/factors/test_analyzer.py`

**Step 1: Write the failing test**

```python
def test_ensure_data_prepared_reprepare_when_period_missing(...):
    # 先 prepare_data(periods=[1])，再請求 periods=[1, 5]
    # 應觸發 re-prepare 而非直接爆 ValueError
    ...
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/factors/test_analyzer.py::test_ensure_data_prepared_reprepare_when_period_missing -v`
Expected: FAIL，顯示缺少 `period_5` 欄位或未重建資料。

**Step 3: Write minimal implementation**

```python
if not hasattr(self, "_clean_data") or (
    periods and any(f"period_{p}" not in self._clean_data.columns for p in periods)
):
    self.prepare_data(periods=periods, price_col=price_col)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/factors/test_analyzer.py::test_ensure_data_prepared_reprepare_when_period_missing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/factors/test_analyzer.py src/factorium/factors/analyzer.py
git commit -m "fix(analyzer): re-prepare data when requested periods are missing"
```

### Task 4: tags 全市場抓取效能風險的防呆/說明

**Files:**
- Modify: `src/factorium/universe/tags.py`
- Test: `tests/universe/test_tags.py`
- Optional docs: `docs/user-guide/`（若對外行為有明確限制）

**Step 1: Decide minimal policy**

在下列二選一中擇一（優先 A，保持相容）：
- A: 補強 docstring 與 warning，明確標示未傳 `symbols` 時可能很慢。
- B: 強制需要 `symbols`（breaking change，需更新文件與測試）。

**Step 2: Write failing test for chosen policy**

```python
def test_fetch_async_warns_when_symbols_none(...):
    ...
```

**Step 3: Implement minimal change**

```python
if symbols is None:
    logger.warning("Fetching tags for all symbols may take a long time...")
```

**Step 4: Run tests**

Run: `pytest tests/universe/test_tags.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/universe/test_tags.py src/factorium/universe/tags.py
git commit -m "docs(universe): document full-market tags fetch latency risk"
```

### Task 5: 清理 vectorized mask 冗餘邏輯

**Files:**
- Modify: `src/factorium/backtest/vectorized.py:173`
- Test: `tests/backtest/test_vectorized.py`

**Step 1: Write failing/regression test**

```python
def test_calculate_weights_masked_assets_remain_zero_after_neutralize(...):
    ...
```

**Step 2: Run test to verify behavior baseline**

Run: `pytest tests/backtest/test_vectorized.py::test_calculate_weights_masked_assets_remain_zero_after_neutralize -v`
Expected: 先建立基準；若測試已過，改成 lock current behavior regression test。

**Step 3: Remove redundant mask re-application**

```python
if self._mask is not None:
    df = df.drop("_masked_signal")
```

**Step 4: Run tests to verify no regression**

Run: `pytest tests/backtest/test_vectorized.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/backtest/test_vectorized.py src/factorium/backtest/vectorized.py
git commit -m "refactor(backtest): remove redundant mask application in weight calc"
```

### Task 6: Final verification and integration checks

**Files:**
- No code change expected

**Step 1: Run targeted test suite**

Run: `pytest tests/data/test_timestamp_utils.py tests/universe/test_metadata.py tests/universe/test_tags.py tests/factors/test_analyzer.py tests/backtest/test_vectorized.py -v`
Expected: PASS

**Step 2: Run broad safety regression**

Run: `pytest tests/factors/test_safe_operations.py -v`
Expected: PASS

**Step 3: Optional full test run (if CI cost acceptable)**

Run: `pytest -q`
Expected: PASS（或僅已知 flaky 測試需註記）

**Step 4: Squash strategy decision (optional)**

保持小步提交（建議），若要合併前整理歷史，於 PR 階段處理，不在實作中途改寫歷史。
