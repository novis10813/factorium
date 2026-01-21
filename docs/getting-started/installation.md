# 安裝

## 使用 pip

```bash
pip install factorium
```

## 使用 uv（推薦）

```bash
uv add factorium
```

## 從原始碼安裝

```bash
git clone https://github.com/novis/factorium.git
cd factorium
uv sync
```

## 依賴套件

Factorium 需要 Python 3.11 或更高版本，主要依賴：

| 套件 | 用途 |
|------|------|
| pandas | 資料處理 |
| numpy | 數值運算 |
| numba | JIT 加速 |
| matplotlib | 圖表繪製 |
| aiohttp | 異步 HTTP（資料下載） |

## 驗證安裝

```python
import factorium
print(factorium.__version__)
```
