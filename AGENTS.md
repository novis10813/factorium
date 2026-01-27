# 程式碼庫慣例與模式

## 專案概述

Factorium 是一個量化因子分析與回測框架，主要模組：

| 模組 | 說明 |
|------|------|
| `factors/` | 因子核心（運算子、解析器、分析器） |
| `data/` | 資料下載與載入 |
| `backtest/` | 回測引擎 |
| `bar.py` | Bar 聚合（Time/Tick/Volume/Dollar） |
| `aggbar.py` | 多標的資料容器 |

---

## `safe_` 函數模式

在此專案中，以 `safe_` 開頭的函數（例如：`safe_mean`, `safe_sum`, `safe_div`）旨在確保計算的「嚴格性」與「安全性」，這對於金融因子的計算尤為重要。

### 共同特點

1.  **嚴格的缺失值 (NaN) 傳遞**:
    *   與標準 Pandas/Numpy 操作（通常會忽略 NaN）不同，這些函數在輸入窗口中包含**任何** `NaN` 或資料長度不足 (`len(x) < window`) 時，會直接回傳 `np.nan`。
    *   這可以防止因數據不完整而產生的錯誤訊號。

2.  **安全性檢查**:
    *   **避免除以零**: 如 `safe_div` 等函數會檢查分母是否為零，以避免產生 `inf` 或導致程式崩潰。
    *   **資料充裕度檢查**: 如 `safe_corr` 會在計算前確認是否有足夠的有效數據點（例如：多於 2 個）。

3.  **safe_div 一致性規範**:
    *   **閾值**: 使用 `POSITION_EPSILON`（`1e-10`）判斷分母接近 0 的情況。
    *   **缺失值回傳**: Pandas 路徑回傳 `np.nan`，Polars 路徑回傳 `null`（建議使用 `pl.lit(None)`）。
    *   **語義**: 分母為 0 或 `abs(denominator) <= POSITION_EPSILON` 時視為缺失，避免產生 `inf`。

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
