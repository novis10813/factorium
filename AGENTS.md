# 程式碼庫慣例與模式

## `safe_` 函數模式

在此專案中，以 `safe_` 開頭的函數（例如：`safe_mean`, `safe_sum`, `safe_div`）旨在確保計算的「嚴格性」與「安全性」，這對於金融因子的計算尤為重要。

### 共同特點

1.  **嚴格的缺失值 (NaN) 傳遞**:
    *   與標準 Pandas/Numpy 操作（通常會忽略 NaN）不同，這些函數在輸入窗口中包含**任何** `NaN` 或資料長度不足 (`len(x) < window`) 時，會直接回傳 `np.nan`。
    *   這可以防止因數據不完整而產生的錯誤訊號。

2.  **安全性檢查**:
    *   **避免除以零**: 如 `safe_div` 等函數會檢查分母是否為零，以避免產生 `inf` 或導致程式崩潰。
    *   **資料充裕度檢查**: 如 `safe_corr` 會在計算前確認是否有足夠的有效數據點（例如：多於 2 個）。

### 範例
```python
def safe_mean(x: pd.Series) -> float:
    # 如果有任何值為 NaN 或長度不足，則回傳 NaN
    return np.nan if (x.isna().any() or len(x) < window) else x.mean()
```
