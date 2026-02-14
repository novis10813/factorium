# Universe User Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 補齊 universe/checklist 的官方使用文件，讓使用者能從文件直接完成「建立資產池 -> 套 mask -> 因子計算 -> 回測」流程。

**Architecture:** 以 `docs/user-guide/universe.md` 作為主入口，並在既有 `bar/factor/backtest/index` 做最小必要串接。先用文件驗收測試定義必要章節與關鍵 API，再逐步補內容，最後以 MkDocs build 驗證可發布品質。

**Tech Stack:** MkDocs + Material、Markdown、pytest（文件驗收測試）

---

### Task 1: 建立文件驗收測試骨架

**Files:**
- Create: `tests/docs/test_universe_user_docs.py`
- Test: `tests/docs/test_universe_user_docs.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_universe_guide_file_exists():
    assert Path("docs/user-guide/universe.md").exists()


def test_universe_guide_has_required_sections():
    text = Path("docs/user-guide/universe.md").read_text(encoding="utf-8")
    required = [
        "# Universe 與 Checklist",
        "## 快速流程",
        "## 與 AggBar 整合",
        "## 與 Factor 整合",
        "## 與 Backtest 整合",
        "## 常見錯誤",
    ]
    for section in required:
        assert section in text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/docs/test_universe_user_docs.py -v`
Expected: FAIL（`docs/user-guide/universe.md` 尚不存在）

**Step 3: Write minimal implementation**

建立 `docs/user-guide/universe.md`，先放最小標題骨架以滿足測試。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/docs/test_universe_user_docs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/docs/test_universe_user_docs.py docs/user-guide/universe.md
git commit -m "test(docs): add universe user-guide acceptance checks"
```

### Task 2: 完成 Universe 主文件內容

**Files:**
- Modify: `docs/user-guide/universe.md`
- Test: `tests/docs/test_universe_user_docs.py`

**Step 1: Write the failing test**

在 `tests/docs/test_universe_user_docs.py` 增加關鍵片段驗證：
- `AggBar.with_mask(`
- `Factor.eval(..., mask=`
- `Backtester(..., mask=`（以專案實際 API 命名為準）

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/docs/test_universe_user_docs.py -v`
Expected: FAIL（關鍵片段尚未補齊）

**Step 3: Write minimal implementation**

在 `docs/user-guide/universe.md` 補齊：
- 概念：universe 與 checklist 的差異
- 快速流程：資料 -> 遮罩 -> 因子 -> 回測
- 可直接執行的最小程式片段
- 常見錯誤（index 對齊、look-ahead、NaN/mask 傳遞）

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/docs/test_universe_user_docs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/user-guide/universe.md tests/docs/test_universe_user_docs.py
git commit -m "docs(universe): add complete universe/checklist user guide"
```

### Task 3: 串接既有文件與導覽

**Files:**
- Modify: `mkdocs.yml`
- Modify: `docs/index.md`
- Modify: `docs/user-guide/bar.md`
- Modify: `docs/user-guide/factor.md`
- Modify: `docs/user-guide/backtest.md`
- Test: `tests/docs/test_universe_user_docs.py`

**Step 1: Write the failing test**

在 `tests/docs/test_universe_user_docs.py` 新增：
- `mkdocs.yml` 導覽含 `user-guide/universe.md`
- `docs/index.md` 含 universe/checklist 入口描述
- `bar/factor/backtest` 各至少一處連到 universe 使用方式

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/docs/test_universe_user_docs.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

更新 `mkdocs.yml` 導覽，並在上述文件加入最小必要交叉連結與示例段落。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/docs/test_universe_user_docs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mkdocs.yml docs/index.md docs/user-guide/bar.md docs/user-guide/factor.md docs/user-guide/backtest.md tests/docs/test_universe_user_docs.py
git commit -m "docs: wire universe guide into nav and related user guides"
```

### Task 4: 發布前驗證

**Files:**
- Modify: `docs/user-guide/universe.md`（若 build 警告需微調）

**Step 1: Write the failing test**

無新增程式測試；以文件建置作為驗收門檻。

**Step 2: Run test to verify it fails**

Run: `uv run mkdocs build`
Expected: 若有 broken links 或 nav 問題則 FAIL

**Step 3: Write minimal implementation**

修正連結、標題層級、程式片段語法標記，直到 build 無錯。

**Step 4: Run test to verify it passes**

Run: `uv run mkdocs build`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/ mkdocs.yml
git commit -m "chore(docs): pass mkdocs build for universe documentation"
```

### Task 5: PR 自我檢查

**Files:**
- Modify: `docs/plans/2026-02-13-universe-docs-pr-plan.md`（若需補充實際執行備註）

**Step 1: Write the failing test**

建立自我檢查清單（人工）：
- [ ] 新頁面在導覽可見
- [ ] 主流程程式片段可讀且命名一致
- [ ] 交叉連結完整

**Step 2: Run test to verify it fails**

人工審閱，任何一項未滿足即視為 FAIL。

**Step 3: Write minimal implementation**

補齊缺漏內容。

**Step 4: Run test to verify it passes**

再次人工審閱 + `uv run mkdocs build`。

**Step 5: Commit**

```bash
git add docs/
git commit -m "docs: finalize universe documentation PR checklist"
```
