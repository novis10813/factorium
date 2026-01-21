# 時間序列回歸運算子 (Time-Series Regression Operators) 實作文件

本文件記錄了 `ts_beta`、`ts_alpha` 和 `ts_resid` 運算子的設計與實作細節。這些運算子用於對兩個因子進行滾動窗口線性回歸分析。

## 1. 概述

我們在 `Factorium` 庫中實作了以下三個時間序列回歸運算子：

*   `ts_beta(other, window)`: 計算滾動窗口內的斜率係數 $\beta$。
*   `ts_alpha(other, window)`: 計算滾動窗口內的截距 $\alpha$。
*   `ts_resid(other, window)`: 計算當期（窗口最後一期）的殘差 $\epsilon$。

這些運算子被添加到了 `TimeSeriesOpsMixin` 中，並透過 `operators.py` 暴露為函數式接口，支援表達式解析。

## 2. 設計決策

### 2.1 底層實作
我們利用了 `TimeSeriesOpsMixin` 中現有的 `ts_cov`、`ts_mean` 和 `ts_std` 來實作回歸運算。這確保了以下一致性：
*   **滾動窗口邏輯**：使用相同的 Pandas rolling 機制。
*   **嚴格的缺失值處理**：窗口內若有任何 `NaN`，結果即為 `NaN`。
*   **資料對齊**：透過嚴格的時間戳記與 Symbol 合併來確保資料對齊。

### 2.2 運算子定義與公式

#### `ts_beta(self, other, window)`
*   **公式**: $\beta = \frac{Cov(self, other)}{Var(other)}$
*   **實作**: `self.ts_cov(other, window) / (other.ts_std(window) ** 2)`
*   **安全性**: 包含除以零的保護機制（處理 `inf`）。

#### `ts_alpha(self, other, window)`
*   **公式**: $\alpha = \bar{self} - \beta \times \bar{other}$
*   **實作**: `self.ts_mean(window) - beta * other.ts_mean(window)`

#### `ts_resid(self, other, window)`
*   **公式**: $\epsilon_t = self_t - (\alpha_t + \beta_t \times other_t)$
*   **實作**: `self - (alpha + beta * other)`
*   **注意**: 此運算使用 `self` 和 `other` 在當期（時間 $t$）的原始值進行計算。

## 3. 重構與優化

在實作過程中，我們對 `src/factorium/factors/operators.py` 進行了重構：
*   **移除冗餘代碼**: 刪除了 `_apply_binary_op` 輔助函數。
*   **純包裝器模式**: 將 `ts_beta`、`ts_alpha` 和 `ts_resid` 實作為純粹的包裝器，直接調用 `Factor` 實例的方法 (例如 `return factor.ts_beta(other, window)`)。這使得 `operators.py` 更加簡潔且符合 Pythonic 風格。

## 4. 測試策略

我們採用了多層次的測試策略來驗證實作的正確性：

### 4.1 單元測試 (`tests/mixins/test_ts_ops.py`)
*   **正確性驗證**:
    *   完美相關 ($y=2x$) $\rightarrow$ $\beta=2, \alpha=0, \epsilon=0$
    *   已知偏移 ($y=x+5$) $\rightarrow$ $\beta=1, \alpha=5$
    *   殘差計算 ($y=2x+\epsilon$)
*   **NaN 處理**: 驗證當窗口內包含 `NaN` 時，輸出嚴格為 `NaN`。

### 4.2 表達式解析器測試 (`tests/test_expression_parser.py`)
*   新增了針對 `ts_beta`、`ts_alpha` 和 `ts_resid` 的測試用例。
*   驗證字串表達式（如 `"ts_beta(close, open, 20)"`）能否正確解析並產生與方法鏈調用相同的結果。

## 5. 檔案變更列表

*   `src/factorium/factors/mixins/ts_ops.py`: 新增運算子實作。
*   `src/factorium/factors/operators.py`: 新增並重構函數式接口。
*   `tests/mixins/test_ts_ops.py`: 新增單元測試。
*   `tests/test_expression_parser.py`: 新增表達式測試。
