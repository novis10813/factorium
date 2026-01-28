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
from factorium import ResearchSession
import polars as pl

# 1. 建立研究 Session（從 Parquet 檔案）
session = ResearchSession.from_parquet("data/btc_1h.parquet")

# 2. 建立簡單動量因子
close = session.factor("close")
momentum = (close.ts_delta(20) / close.ts_shift(20)).cs_rank()

# 3. 一行完成分析與回測（簡易報告）
print(session.quick_report(momentum))
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
| [因子分析](user-guide/analyzer.md) | IC / 分層收益等分析工具 |
| [策略回測](user-guide/backtest.md) | 向量化回測與權重約束 |

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
