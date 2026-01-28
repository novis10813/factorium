# Factorium

**量化因子分析與回測框架**

Factorium 是一個專為量化研究設計的 Python 函式庫，提供高效能的因子計算、分析與回測工具。

---

## 核心特色

<div class="grid cards" markdown>

-   :material-chart-line: **因子運算**

    ---

    完整的時間序列 (TS)、橫截面 (CS) 與數學運算子，支援鏈式操作

-   :material-database: **資料處理**

    ---

    支援多種 Bar 聚合方式 (Time/Tick/Volume/Dollar)，內建 Binance 數據下載器

-   :material-test-tube: **因子分析**

    ---

    IC 分析、分位數回報、換手率計算等完整評估工具

-   :material-chart-box: **策略回測**

    ---

    基於因子的回測引擎，支援市場中性與 Long-only 策略

</div>

---

## 快速開始

```python
from factorium import AggBar, Factor
from factorium.backtest import Backtester
import polars as pl

# 載入數據（從 Parquet 檔案）
agg = AggBar.from_df(pl.read_parquet("data/btc_1h.parquet"))

# 計算動量因子
momentum = (agg["close"] / agg["close"].ts_shift(20) - 1).cs_rank()

# 執行回測
bt = Backtester(prices=agg, signal=momentum, neutralization="market")
result = bt.run()

print(result.metrics)
```

---

## 安裝

```bash
pip install factorium
```

或使用 uv：

```bash
uv add factorium
```

---

## 文檔導覽

| 章節 | 說明 |
|------|------|
| [快速開始](getting-started/quickstart.md) | 五分鐘上手教學 |
| [資料獲取](getting-started/data-acquisition.md) | 下載與載入市場數據 |
| [Bar 聚合](user-guide/bar.md) | 不同類型的 K 線聚合 |
| [Factor 因子](user-guide/factor.md) | 因子計算與運算子 |
| [策略回測](user-guide/backtest.md) | 因子回測系統 |

---

## 專案結構

```
factorium/
├── data/          # 資料下載與載入
├── factors/       # 因子核心 (運算子、解析器、分析器)
├── backtest/      # 回測引擎
├── bar.py         # Bar 聚合
└── aggbar.py      # 多標的資料容器
```
