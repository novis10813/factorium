# Factorium

量化因子研究與回測工具庫（Polars-first, research-first）。

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 特色一覽

- **資料管線**：`BinanceDataLoader` + `AggBar`，一行從 Binance 歷史資料到多標的 OHLCV Panel。
- **因子引擎**：Polars 驅動的 `Factor`，提供時間序列 / 橫截面 / 數學運算與表達式解析 (`FactorExpressionParser`)。
- **因子分析**：`FactorAnalyzer` + `FactorAnalysisResult`，支援 IC、分層收益、累積收益與圖表。
- **向量化回測**：`VectorizedBacktester`（經由 `factorium.backtest.Backtester` 暴露），Polars 向量化計算 + 完整績效指標。
- **研究工作流**：`ResearchSession` + `FactorReport`，把「載入 → 建因子 → 分析 → 回測 → 報告」收斂成幾行程式碼。

完整文件請參考 `docs/` 或線上文檔（若已部署）。

---

## 安裝

```bash
# 推薦：使用 uv
uv add factorium

# 或使用 pip
pip install factorium
```

開發環境：

```bash
git clone https://github.com/novis10813/factorium.git
cd factorium
uv sync --dev
```

---

## 五分鐘快速開始

### 1. 使用 `ResearchSession` 跑一個簡單策略

```python
from factorium import ResearchSession
import polars as pl

# 從 Parquet 檔案建立 AggBar 並建立研究 Session
session = ResearchSession.from_parquet("data/btc_1h.parquet")

# 建立一個簡單的動量因子
close = session.factor("close")
momentum = (close.ts_delta(20) / close.ts_shift(20)).cs_rank()

# 一行完成分析與回測的文字報告
print(session.quick_report(momentum, periods=1))
```

### 2. 從 Binance 載入多標的資料

```python
from factorium import BinanceDataLoader

loader = BinanceDataLoader()
agg = loader.load_aggbar(
    symbols=["BTCUSDT", "ETHUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=7,
    bar_type="time",
    interval=60_000,  # 1 分鐘（毫秒）
)

print(agg.symbols)
print(agg.cols)
print(agg.to_polars().head())
```

---

## 核心元件概觀

### `BinanceDataLoader` & `AggBar`

- `BinanceDataLoader.load_aggbar(...)`：下載 + 快取 + 聚合，多標的、多種 bar 類型（time / tick / volume / dollar）。
- `AggBar`：多標的長表 Panel，欄位至少包含 `start_time`, `end_time`, `symbol`, `open`, `high`, `low`, `close`, `volume`。

常見用法：

```python
from factorium import AggBar
import polars as pl

agg = AggBar.from_df(pl.read_parquet("data/aggregated.parquet"))
close = agg["close"]      # 回傳 Factor
volume = agg["volume"]
```

更完整說明見：`docs/user-guide/bar.md`、`docs/user-guide/factor.md`。

### `Factor` 與運算子

`Factor` 是 Factorium 的核心資料結構，代表「多標的時間序列因子」，內部使用 Polars `LazyFrame` 做運算。

範例：

```python
close = agg["close"]

# 價格動量
momentum = close.ts_delta(20) / close.ts_shift(20)

# 波動調整後動量
volatility = close.ts_std(20)
risk_adj = (close.ts_delta(20) / close.ts_shift(20)) / volatility

# 橫截面排名
signal = risk_adj.cs_rank()
```

你也可以用字串表達式定義因子（適合從設定檔載入）：

```python
from factorium.factors import FactorExpressionParser

parser = FactorExpressionParser()
ctx = {"close": agg["close"], "volume": agg["volume"]}

factor = parser.parse("ts_delta(close, 20) / ts_shift(close, 20)", context=ctx)
```

### 因子分析：`FactorAnalyzer` & `FactorAnalysisResult`

```python
from factorium.factors import FactorAnalyzer

analyzer = FactorAnalyzer(factor=signal, prices=agg)
result = analyzer.analyze(price_col="close", periods=1)

print(result.ic_summary)         # dict: mean_ic, ic_std, ic_ir, t-stat
print(result.quantile_returns)   # 分層收益
```

### 向量化回測：`Backtester` / `VectorizedBacktester`

```python
from factorium.backtest import Backtester

bt = Backtester(
    prices=agg,
    signal=signal,
    initial_capital=10_000.0,
    neutralization="market",   # "market" 或 "none"
    transaction_cost=0.0003,
    frequency="1h",
)
backtest_result = bt.run()

print(backtest_result.metrics)   # Sharpe, Sortino, Calmar, max_drawdown, win_rate, ...
```

詳細說明見：`docs/user-guide/backtest.md`。

### 研究工作流：`ResearchSession` & `FactorReport`

`ResearchSession` 封裝了一般研究流程：

```python
from factorium import ResearchSession
from factorium.research import FactorReport

session = ResearchSession.from_parquet("data/multi_symbol.parquet")
factor = session.create_factor("ts_delta(close, 20) / ts_shift(close, 20)", name="momentum")

# 1. 分析 + 回測 + 報告物件
report = FactorReport.generate(session, factor)
print(report.summary())

# 2. 或使用 session.quick_report 取得文字摘要
print(session.quick_report(factor, periods=1))
```

---

## 測試

本專案使用 `pytest`，建議透過 uv 執行：

```bash
# 全部測試
uv run pytest

# 指定模組
uv run pytest tests/factors/test_analyzer_polars.py

# 顯示覆蓋率
uv run pytest --cov=factorium
```

更多測試策略與設計細節請參考 `docs/dev/testing.md`。

---

## 授權與作者

- 授權：MIT License（詳見 `LICENSE`）
- 作者：Samuel Chang

