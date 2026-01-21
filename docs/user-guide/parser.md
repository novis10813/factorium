## 因子表達式解析器 (`factorium/factors/parser.py`)

## 概述

`FactorExpressionParser` 提供了一個功能強大的表達式解析器，允許使用字符串表達式來構建因子，類似於 alpha101 的風格。  
這使得因子構建更加靈活，特別適合從配置文件或外部系統讀取因子定義。

### 主要特性

- **函數調用**：`ts_delta(close, 20)`、`rank(momentum)` 等
- **變數引用**：`close`、`volume`（從 context 解析）
- **數值常數**：整數和浮點數（如 `20`、`3.14`）
- **二元運算子**：`+`、`-`、`*`、`/`（支援運算子優先順序）
- **括號**：`(expression)` 用於控制運算順序

---

## 基本用法

### 使用 `Factor.from_expression`

最簡單的方式是透過 `Factor.from_expression` 類方法：

```python
from factorium import Factor, AggBar

agg = loader.load_aggbar(...)
close = agg['close']
volume = agg['volume']

# 解析表達式並建立因子
momentum = Factor.from_expression(
    "ts_delta(close, 20) / ts_shift(close, 20)",
    context={'close': close}
)

# 複雜表達式
complex_factor = Factor.from_expression(
    "(close + volume) * 2 - ts_mean(close, 10)",
    context={'close': close, 'volume': volume}
)
```

### 直接使用 `FactorExpressionParser`

如果需要更細緻的控制，可以直接使用解析器：

```python
from factorium.factors.parser import FactorExpressionParser

parser = FactorExpressionParser()
result = parser.parse(
    "ts_delta(close, 20) / ts_shift(close, 20)",
    context={'close': close}
)
```

---

## 表達式語法

### 函數調用

支援所有 Factor 運算子，包括：

- **時間序列運算子**：`ts_delta`、`ts_mean`、`ts_std`、`ts_zscore` 等
- **橫截面運算子**：`rank`、`mean`、`median`
- **數學運算子**：`abs`、`log`、`sqrt`、`pow` 等

```python
# 時間序列運算子
momentum = Factor.from_expression(
    "ts_delta(close, 20) / ts_shift(close, 20)",
    context={'close': close}
)

zscore = Factor.from_expression(
    "ts_zscore(close, 20)",
    context={'close': close}
)

# 橫截面運算子
ranked = Factor.from_expression(
    "rank(ts_mean(close, 10))",
    context={'close': close}
)

# 數學運算子
log_factor = Factor.from_expression(
    "log(close)",
    context={'close': close}
)
```

### 變數引用

變數名稱必須在 `context` 字典中定義：

```python
close = agg['close']
volume = agg['volume']
open_price = agg['open']

# 使用多個變數
factor = Factor.from_expression(
    "(close - open) / volume",
    context={'close': close, 'open': open_price, 'volume': volume}
)
```

### 數值常數

支援整數和浮點數：

```python
# 整數
factor = Factor.from_expression(
    "close + 10",
    context={'close': close}
)

# 浮點數
factor = Factor.from_expression(
    "close * 2.5",
    context={'close': close}
)

# 負數
factor = Factor.from_expression(
    "add(close, -10)",
    context={'close': close}
)

# 科學記號（支援）
factor = Factor.from_expression(
    "mul(close, 1e-3)",
    context={'close': close}
)
```

### 二元運算子

支援四則運算，運算子優先順序遵循標準數學規則：

- `*`、`/` 優先於 `+`、`-`
- 使用括號可以改變運算順序

```python
# 加法
factor = Factor.from_expression(
    "close + volume",
    context={'close': close, 'volume': volume}
)

# 減法
factor = Factor.from_expression(
    "close - open",
    context={'close': close, 'open': open_price}
)

# 乘法
factor = Factor.from_expression(
    "close * 2",
    context={'close': close}
)

# 除法
factor = Factor.from_expression(
    "close / volume",
    context={'close': close, 'volume': volume}
)

# 運算子優先順序
# close + open * 2 等同於 close + (open * 2)
factor = Factor.from_expression(
    "close + open * 2",
    context={'close': close, 'open': open_price}
)

# 使用括號改變順序
# (close + open) * 2
factor = Factor.from_expression(
    "(close + open) * 2",
    context={'close': close, 'open': open_price}
)
```

### 括號

括號用於控制運算順序：

```python
# 複雜表達式
factor = Factor.from_expression(
    "(close + open) * 2 / (volume + 1)",
    context={'close': close, 'open': open_price, 'volume': volume}
)
```

---

## 完整範例

### 範例 1：動量因子

```python
from factorium import Factor, BinanceDataLoader

loader = BinanceDataLoader()
agg = loader.load_aggbar(
    symbols=["BTCUSDT", "ETHUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=30,
    timestamp_col="transact_time",
    price_col="price",
    volume_col="quantity",
    interval_ms=60_000
)

close = agg['close']

# 使用表達式構建動量因子
momentum = Factor.from_expression(
    "ts_delta(close, 20) / ts_shift(close, 20)",
    context={'close': close}
)

# 等價於方法鏈式調用
momentum_chain = close.ts_delta(20) / close.ts_shift(20)
```

### 範例 2：複雜因子組合

```python
close = agg['close']
volume = agg['volume']

# 構建複雜因子：波動調整的動量排名
complex_factor = Factor.from_expression(
    "rank(div(ts_delta(close, 20), ts_std(close, 20)))",
    context={'close': close}
)

# 等價於
complex_chain = (close.ts_delta(20) / close.ts_std(20)).rank()
```

### 範例 3：使用中綴運算子

```python
close = agg['close']
open_price = agg['open']
volume = agg['volume']

# 中綴運算子語法（更直觀）
factor = Factor.from_expression(
    "(close - open) * volume / ts_mean(volume, 20)",
    context={'close': close, 'open': open_price, 'volume': volume}
)
```

### 範例 4：從配置文件讀取

```python
import json

# 從配置文件讀取因子定義
with open('factors.json', 'r') as f:
    factor_configs = json.load(f)

# 載入資料
agg = loader.load_aggbar(...)
close = agg['close']
volume = agg['volume']

# 建立 context
context = {
    'close': close,
    'volume': volume,
}

# 批量建立因子
factors = {}
for name, expr in factor_configs.items():
    factors[name] = Factor.from_expression(expr, context=context)
```

---

## 錯誤處理

### 未定義變數

如果表達式中使用了未在 `context` 中定義的變數，會拋出 `ValueError`：

```python
try:
    factor = Factor.from_expression(
        "ts_delta(unknown_var, 20)",
        context={'close': close}
    )
except ValueError as e:
    print(f"Error: {e}")  # Undefined variable: unknown_var
```

### 未知函數

如果使用了不存在的函數，會拋出 `ValueError`：

```python
try:
    factor = Factor.from_expression(
        "unknown_func(close, 20)",
        context={'close': close}
    )
except ValueError as e:
    print(f"Error: {e}")  # Unknown function: unknown_func
```

### 語法錯誤

如果表達式語法不正確，會拋出 `ValueError`：

```python
try:
    factor = Factor.from_expression(
        "ts_delta(close, 20",  # 缺少右括號
        context={'close': close}
    )
except ValueError as e:
    print(f"Error: {e}")  # Failed to parse expression...
```

---

## 與方法鏈式調用的對應關係

表達式解析器與 Factor 的方法鏈式調用完全等價：

| 表達式 | 方法鏈式調用 |
|--------|------------|
| `ts_delta(close, 20)` | `close.ts_delta(20)` |
| `ts_mean(close, 10)` | `close.ts_mean(10)` |
| `rank(close)` | `close.rank()` |
| `abs(close)` | `close.abs()` |
| `close + open` | `close + open` |
| `close * 2` | `close * 2` |
| `ts_delta(close, 20) / ts_shift(close, 20)` | `close.ts_delta(20) / close.ts_shift(20)` |
| `rank(ts_mean(close, 10))` | `close.ts_mean(10).rank()` |

---

## 技術細節

### 解析器實作

`FactorExpressionParser` 使用 `pyparsing` 庫來構建解析器：

- **語法分析**：使用 `infix_notation` 處理運算子優先順序
- **遞迴下降**：支援嵌套函數調用和括號表達式
- **類型推斷**：自動識別變數、函數、數值和運算子

### 支援的運算子優先順序

1. `*`、`/`（左結合）
2. `+`、`-`（左結合）

### 函數參數

所有函數調用的參數都會被遞迴評估，可以是：
- 另一個函數調用
- 變數引用
- 數值常數
- 表達式結果

---

## 小結

- `FactorExpressionParser` 提供靈活的字符串表達式解析功能
- 支援所有 Factor 運算子（時間序列、橫截面、數學運算）
- 支援中綴運算子語法，更符合直覺
- 與方法鏈式調用完全等價，可互相轉換
- 適合從配置文件或外部系統讀取因子定義

透過表達式解析器，可以更靈活地構建和管理因子，特別適合需要動態配置因子的場景。

