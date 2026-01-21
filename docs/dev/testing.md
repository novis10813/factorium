# Pytest 策略與雙向驗證

本專案使用**雙向驗證**策略來驗證 `Factor` 數學運算的正確性。

## 雙向驗證概念

核心想法是驗證高階的 `Factor` 運算（封裝了 `pandas` 和 `numpy` 邏輯）產生的結果與直接在底層 `pandas.DataFrame` 上應用對應的 `numpy` 運算完全一致。

*   **正向路徑（實作）**：在 `Factor` 物件上執行方法（例如 `f1.add(f2)`）。這使用了 `MathOpsMixin` 實作，包含合併邏輯和錯誤處理。
*   **反向路徑（驗證）**：手動從 `Factor` 物件中提取數據，並應用預期的 `numpy` 函數（例如 `np.add(f1.data, f2.data)`）。
*   **斷言**：使用 `pd.testing.assert_frame_equal`（或等效的序列對齊方法）比較兩個結果 DataFrame，確保它們在浮點數容差範圍內相同。

## 實作細節

驗證邏輯封裝在 `tests/mixins/test_mathmixin.py` 中。

### 輔助函數
*   `emulate_unary_op(factor, func)`：對因子的數據應用 numpy 函數。
*   `emulate_binary_op(f1, f2, func)`：根據 `start_time`、`end_time`、`symbol` 合併兩個因子並應用函數。
*   `assert_factor_equals_df(factor_res, expected_series)`：對齊並比較結果。

### 範例
```python
def test_add_factor(factor_close, factor_open):
    # 正向：類別方法
    res = factor_close + factor_open
    
    # 反向：手動模擬
    expected = emulate_binary_op(factor_close, factor_open, lambda x, y: x + y)
    
    # 驗證
    assert_factor_equals_df(res, expected)
```

## 執行測試

執行數學混入測試：
```bash
uv run pytest tests/mixins/test_mathmixin.py
```

執行所有測試：
```bash
uv run pytest
```
