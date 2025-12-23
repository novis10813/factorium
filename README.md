# Factorium

## 專案說明
量化因子分析與研究工具庫。

## 測試

本專案使用 `pytest` 進行測試。

### 執行測試
執行所有測試：
```bash
uv run pytest
```

執行特定測試檔案：
```bash
uv run pytest tests/mixins/test_mathmixin.py
```

更多關於測試策略的細節，特別是數學運算子的「雙向驗證」流程，請參閱 [docs/pytest.md](docs/pytest.md)。
