# 程式碼庫慣例與模式

## 專案概述

Factorium 是一個量化因子分析與回測框架，主要模組：

| 模組 | 說明 |
|------|------|
| `factors/` | 因子核心（運算子、解析器、分析器） |
| `data/` | 資料下載與載入 |
| `backtest/` | 回測引擎 |
| `aggbar.py` | 多標的資料容器 |

---

## `safe_` 函數模式

在此專案中，以 `safe_` 開頭的函數（例如：`safe_mean`, `safe_sum`, `safe_div`）旨在確保計算的「嚴格性」與「安全性」，這對於金融因子的計算尤為重要。

### 共同特點

1. **嚴格的缺失值 (NaN) 傳遞**:
    * 與標準 Pandas/Numpy 操作（通常會忽略 NaN）不同，這些函數在輸入窗口中包含**任何** `NaN` 或資料長度不足 (`len(x) < window`) 時，會直接回傳 `np.nan`。
    * 這可以防止因數據不完整而產生的錯誤訊號。

2. **安全性檢查**:
    * **避免除以零**: 如 `safe_div` 等函數會檢查分母是否為零，以避免產生 `inf` 或導致程式崩潰。
    * **資料充裕度檢查**: 如 `safe_corr` 會在計算前確認是否有足夠的有效數據點（例如：多於 2 個）。

3. **safe_div 一致性規範**:
    * **閾值**: 使用 `POSITION_EPSILON`（`1e-10`）判斷分母接近 0 的情況。
    * **缺失值回傳**: Pandas 路徑回傳 `np.nan`，Polars 路徑回傳 `null`（建議使用 `pl.lit(None)`）。
    * **語義**: 分母為 0 或 `abs(denominator) <= POSITION_EPSILON` 時視為缺失，避免產生 `inf`。

### 範例

```python
def safe_mean(x: pd.Series) -> float:
    # 如果有任何值為 NaN 或長度不足，則回傳 NaN
    return np.nan if (x.isna().any() or len(x) < window) else x.mean()
```

---

## Backtest 模組常數

| 常數 | 值 | 用途 |
|------|-----|------|
| `POSITION_EPSILON` | `1e-10` | 判斷持倉變動是否有意義的閾值 |
| `MIN_PERIODS_PER_YEAR` | `1.0` | `periods_per_year` 最小值 |
| `MAX_PERIODS_PER_YEAR` | `~525960` | `periods_per_year` 最大值（分鐘級） |

---

## 文檔結構

文檔使用 MkDocs + Material 主題，結構如下：

```
docs/
├── index.md                    # 首頁
├── getting-started/            # 快速開始
│   ├── installation.md
│   ├── quickstart.md
│   └── data-acquisition.md
├── user-guide/                 # 使用指南
│   ├── bar.md
│   ├── factor.md
│   ├── parser.md
│   ├── analyzer.md
│   └── backtest.md
└── dev/                        # 開發者文檔
    ├── testing.md
    └── regression-operators.md
```

### 本地預覽

```bash
uv run mkdocs serve
```

### 部署到 GitHub Pages

```bash
uv run mkdocs gh-deploy
```

---

## Git 分支策略

本專案採用簡化的 Git Flow 模式，使用兩條長期分支進行開發與發布管理。完整的版本規劃請參考 [GitHub Discussions - Roadmap](https://github.com/novis10813/factorium/discussions/17)。

### 分支定義

#### `main` 分支

**用途**：穩定發布分支（Production Branch）

* **狀態**：隨時可部署的穩定版本
* **內容**：僅包含已正式發布的版本（對應 PyPI 上的公開版本）
* **版本號**：修訂版（patch）發布，如 `0.3.0` → `0.3.1` → `0.3.2`
* **保護規則**：
  * 僅接受來自 `dev` 分支的 PR 或 hotfix 分支的合併
  * 每次合併必須更新 `pyproject.toml` 版本號
  * 合併後必須打 Tag 並發布 GitHub Release（觸發 PyPI 自動發布）

#### `dev` 分支

**用途**：開發整合分支（Development Branch）

* **狀態**：下一個次版本（minor version）的開發分支
* **內容**：新功能開發、重構、非緊急 bug 修復
* **版本號**：對應下一個次版本，如當 `main` 為 `0.3.x` 時，`dev` 開發 `0.4.0`
* **合併來源**：
  * 功能分支（`feat/*`）
  * 重構分支（`refactor/*`）
  * Hotfix 分支（需同步回 `dev` 避免問題重現）

### 分支生命週期示意圖

```
main      ─v0.3.0────v0.3.1──────v0.3.2───  (穩定版)
           │          │           │
dev       ─●──────────●───────────●────────  (開發 0.4.0)
           │          │
hotfix    ─┴──────────┴───────────┘          (緊急修復後合併回 main 和 dev)
```

---

## Hotfix 流程

當 `main` 分支（已發布的穩定版本）發現**嚴重 bug** 時，使用 Hotfix 流程快速修復並發布修訂版。

### 何時使用 Hotfix

**適用情境**（滿足任一即啟動 Hotfix）：

* 計算結果錯誤（如因子數值計算錯誤、回測績效計算錯誤）
* 核心功能失效（如資料載入失敗、無法執行回測）
* 安全性問題或資料遺失風險
* 用戶回報的緊急阻塞性問題

**不適用情境**（應累積到下一個版本）：

* 次要功能缺陷
* 文檔錯誤或範例問題
* 性能優化（非嚴重性能問題）
* 非緊急的 API 改進

### Hotfix 操作步驟

#### 1. 建立 Hotfix 分支

1. 從 `main` 分支建立 hotfix 分支
2. 修復 Bug 並測試
3. 合併回 main 並發布修訂版
4. **關鍵步驟**：同步 Hotfix 回 dev 分支
5. 清理 Hotfix 分支（可選）
6. 在 GitHub 建立 Release

### Hotfix 示意圖

```
main      ─v0.3.2────────────v0.3.3─  (Hotfix: 修復嚴重 bug)
                \            /
hotfix            ●─────────●        (hotfix/issue-18)
                           /
dev       ─────────────────●─────────  (同步 hotfix 避免問題重現)
```

---
