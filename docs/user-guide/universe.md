# Universe 與 Checklist

Universe / Checklist 用來建立資產池遮罩（mask），把因子與回測限制在符合條件的標的集合。

- `Universe`：偏向交易規則與市場結構條件（例如排除穩定幣、上市天數）。
- `Checklist`：偏向研究條件與流動性門檻（例如成交量、流動性、標籤條件）。

## 快速流程

以下流程示範「建立資產池 -> 套 mask -> 因子計算 -> 回測」。

```python
from factorium import (
    Backtester,
    Checklist,
    Factor,
    MinLiquidity,
    MinVolume,
    Universe,
    ExcludeStablecoins,
    load_aggbar,
)

# 1) 載入資料
bar = load_aggbar("BTCUSDT", start="2023-01-01", end="2024-01-01", timeframe="1d")

# 2) 建立 Universe / Checklist
universe = Universe(rules=[ExcludeStablecoins()])
checklist = Checklist(filters=[MinVolume(window=20, threshold=1_000_000), MinLiquidity(window=20, threshold=100_000)])

# metadata/tags 可由 MetadataProvider / TagProvider 產生
metadata = {}
tags = {}

# 3) 產生 mask 欄位
bar = bar.with_mask(name="universe_mask", mask_source=universe, metadata=metadata, tags=tags)
bar = bar.with_mask(name="checklist_mask", mask_source=checklist, metadata=metadata, tags=tags)

# 4) 因子計算使用同一個 mask
factor = Factor("(close / shift(close, 5)) - 1")
result = factor.eval(bar, periods=5, quantiles=5, mask="checklist_mask")

# 5) 回測使用同一個 mask
bt = Backtester(bar.data, signal=result.data, mask="checklist_mask")
stats = bt.run()
```

## 與 AggBar 整合

`AggBar.with_mask()` 會把 `Universe` 或 `Checklist` 的條件計算成布林欄位，直接寫回 `AggBar.data`。

```python
bar = bar.with_mask(
    name="universe_mask",
    mask_source=universe,
    metadata=metadata,
    tags=tags,
)
```

注意：

- `name` 不能覆蓋保留欄位（如 `open`, `close`, `volume`, `symbol`）。
- `metadata` 需包含與 `symbol` 對應的資訊；若規則依賴 `tags`，必須提供 `tags`。

## 與 Factor 整合

`Factor.eval` 可透過 `mask` 參數限制有效樣本：

```python
result = factor.eval(
    bar,
    periods=5,
    quantiles=5,
    mask="checklist_mask",
)
```

這代表因子分位分組與後續分析只在 `checklist_mask == True` 的資料上進行。

## 與 Backtest 整合

`Backtester(...)` 支援 `mask=`，用來限制可持倉資產：

```python
bt = Backtester(
    prices=bar.data,
    signal=result.data,
    holding_period=5,
    mask="checklist_mask",
)
stats = bt.run()
```

當 mask 為 `False` 或 `null` 時，回測會把該資產權重歸零。

## 常見錯誤

- `mask` 欄位名稱拼錯，導致 `Factor.eval(..., mask=...)` 或 `Backtester(..., mask=...)` 找不到欄位。
- `metadata` / `tags` 與 `symbol` 不一致，造成遮罩結果異常。
- 先用完整資料算 signal，再事後套 mask，可能引入 look-ahead；請在計算流程中一致傳入同一個 mask。
- 忽略空值處理，讓 `null` 直接進入決策。建議把 mask 欄位維持明確布林語義（`True` 可交易，其他視為不可交易）。
