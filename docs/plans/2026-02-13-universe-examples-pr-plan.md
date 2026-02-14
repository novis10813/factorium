# Universe Example Notebook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `examples/` 新增 universe/checklist 實戰 notebook，並更新 examples 導覽，讓使用者能照著範例重現遮罩化研究流程。

**Architecture:** 以新 notebook `examples/05_universe_checklist_workflow.ipynb` 作為單一教學入口，README 只保留高層導引。先建立結構驗收測試（檔案存在、README 收錄、必要章節標題），再逐步填入可執行 cell，最後用 nbconvert 執行驗證。

**Tech Stack:** Jupyter Notebook、Python、pytest、nbformat/nbconvert

---

### Task 1: 建立範例驗收測試骨架

**Files:**
- Create: `tests/examples/test_universe_notebook_docs.py`
- Test: `tests/examples/test_universe_notebook_docs.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_universe_notebook_exists():
    assert Path("examples/05_universe_checklist_workflow.ipynb").exists()


def test_examples_readme_mentions_universe_notebook():
    text = Path("examples/README.md").read_text(encoding="utf-8")
    assert "05_universe_checklist_workflow.ipynb" in text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/examples/test_universe_notebook_docs.py -v`
Expected: FAIL（新 notebook 尚未建立，README 尚未更新）

**Step 3: Write minimal implementation**

建立空白 notebook 檔與 README 最小條目。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/examples/test_universe_notebook_docs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/examples/test_universe_notebook_docs.py examples/05_universe_checklist_workflow.ipynb examples/README.md
git commit -m "test(examples): add acceptance checks for universe notebook"
```

### Task 2: 定義 notebook 結構與教學章節

**Files:**
- Modify: `examples/05_universe_checklist_workflow.ipynb`
- Modify: `tests/examples/test_universe_notebook_docs.py`

**Step 1: Write the failing test**

在測試中用 `nbformat` 檢查 notebook markdown 標題至少包含：
- `# Universe Checklist Workflow`
- `## Build Universe and Checklist`
- `## Apply Mask to AggBar`
- `## Evaluate Factor with Mask`
- `## Run Backtest with Mask`

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/examples/test_universe_notebook_docs.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

在 notebook 加入對應章節 markdown cell（先不放完整程式碼）。

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/examples/test_universe_notebook_docs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/05_universe_checklist_workflow.ipynb tests/examples/test_universe_notebook_docs.py
git commit -m "docs(examples): scaffold universe workflow notebook sections"
```

### Task 3: 填入可執行程式流程（最小可重現）

**Files:**
- Modify: `examples/05_universe_checklist_workflow.ipynb`
- Modify: `tests/examples/test_universe_notebook_docs.py`

**Step 1: Write the failing test**

測試新增關鍵 API 片段檢查（以 cell 內容字串比對）：
- `with_mask(`
- `mask=`
- `universe` 或 `checklist` 建立呼叫

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/examples/test_universe_notebook_docs.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

補齊 notebook code cell：
- 載入資料與必要欄位
- 建立 universe/checklist
- 產生並套用 mask 到 `AggBar`
- 執行 `Factor.eval(..., mask=...)`
- 執行回測（含 mask 版本）
- 增加與未套 mask 的最小比較表格

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/examples/test_universe_notebook_docs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/05_universe_checklist_workflow.ipynb tests/examples/test_universe_notebook_docs.py
git commit -m "feat(examples): add executable universe checklist workflow notebook"
```

### Task 4: 更新 examples 導覽與執行驗證

**Files:**
- Modify: `examples/README.md`
- Modify: `examples/05_universe_checklist_workflow.ipynb`（若執行驗證需修正）

**Step 1: Write the failing test**

在 `tests/examples/test_universe_notebook_docs.py` 增加 README 驗收：
- notebook 目的說明
- 前置條件
- 建議執行順序（01 -> ... -> 05）

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/examples/test_universe_notebook_docs.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

更新 `examples/README.md`，補齊第 5 本 notebook 的說明與順序。

**Step 4: Run test to verify it passes**

Run:
- `uv run pytest tests/examples/test_universe_notebook_docs.py -v`
- `uv run python -m jupyter nbconvert --to notebook --execute examples/05_universe_checklist_workflow.ipynb --output /tmp/05_universe_checklist_workflow.executed.ipynb`

Expected: PASS（測試通過，notebook 可從頭執行）

**Step 5: Commit**

```bash
git add examples/README.md examples/05_universe_checklist_workflow.ipynb tests/examples/test_universe_notebook_docs.py
git commit -m "docs(examples): document universe notebook and verify execution"
```

### Task 5: PR 自我檢查

**Files:**
- Modify: `docs/plans/2026-02-13-universe-examples-pr-plan.md`（若需補充執行備註）

**Step 1: Write the failing test**

建立人工清單：
- [ ] Notebook 每節都有文字解說與輸出解讀
- [ ] 程式碼無 look-ahead 寫法
- [ ] README 與 notebook API 名稱一致

**Step 2: Run test to verify it fails**

人工審閱，任一項不滿足即 FAIL。

**Step 3: Write minimal implementation**

修正文案、變數命名與示例片段。

**Step 4: Run test to verify it passes**

再次人工審閱 + 重新執行測試與 nbconvert。

**Step 5: Commit**

```bash
git add examples/ tests/examples/
git commit -m "chore(examples): finalize universe notebook PR checklist"
```
