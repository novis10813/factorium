# 五分鐘教學

本教學將帶你快速上手 Factorium 的核心功能。

---

## 1. 載入數據

```python
from factorium import BinanceDataLoader

# 建立 loader
loader = BinanceDataLoader()

# 載入並聚合成 1 分鐘 K 線（自動下載、快取）
agg = loader.load_aggbar(
    symbols=["BTCUSDT", "ETHUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=7,
    bar_type="time",      # 時間條（也支援 "tick", "volume", "dollar"）
    interval=60_000,      # 1 分鐘（毫秒）
)

print(f"標的數量: {len(agg.symbols)}")
print(f"資料列數: {agg.metadata.num_rows:,}")
```

---

## 2. 查看 AggBar 結構

```python
# AggBar 是多標的 OHLCV 容器
print(f"標的: {agg.symbols}")
print(f"欄位: {agg.cols}")
print(f"時間範圍: {agg.metadata.min_time} ~ {agg.metadata.max_time}")

# 查看資料（Polars DataFrame）
print(agg.to_polars().head())

# 如需 Pandas
print(agg.to_df().head())
```

---

## 3. 計算因子

```python
# 從 AggBar 提取欄位，回傳 Factor 物件
close = agg["close"]
volume = agg["volume"]

# 動量因子：過去 20 期報酬
momentum = close.ts_delta(20) / close.ts_shift(20)

# 波動率因子：過去 20 期標準差（使用百分比變化）
volatility = (close.ts_delta(1) / close.ts_shift(1)).ts_std(20)

# 橫截面排名（0~1 之間）
momentum_rank = momentum.cs_rank()

print(momentum_rank.to_pandas().head())
```

---

## 4. 使用 ResearchSession 一次串好「因子 → 分析 → 回測」

```python
from factorium import ResearchSession

# 建立研究工作階段
session = ResearchSession(agg)

# 建立因子（也可以用 expression string）
momentum = session.create_factor("ts_delta(close, 20) / ts_shift(close, 20)", name="momentum")
signal = momentum.cs_rank()

# 快速分析（IC + 分層收益）
analysis = session.analyze(signal, periods=1)
print(analysis.ic_summary)

# 執行向量化回測（預設使用 VectorizedBacktester）
result = session.backtest(signal, neutralization="market")
print(result.metrics)
```

如需更精簡的文字報告，可以使用：

```python
report_text = session.quick_report(signal, periods=1)
print(report_text)
```

---

## 5. 不同類型的 Bar 聚合

除了時間條，Factorium 也支援其他類型的 bar 聚合：

```python
# Tick Bar：每 1000 筆交易聚合成一個 bar
tick_agg = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=1,
    bar_type="tick",
    interval=1000,  # 1000 筆交易
)

# Volume Bar：每累積 100 BTC 聚合成一個 bar
volume_agg = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=1,
    bar_type="volume",
    interval=100,  # 100 單位成交量
)

# Dollar Bar：每累積 1,000,000 美元聚合成一個 bar
dollar_agg = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=1,
    bar_type="dollar",
    interval=1_000_000,  # 100 萬美元
)
```

---

## 下一步

- [Bar 聚合](../user-guide/bar.md) - 深入了解不同類型的 K 線
- [Factor 因子](../user-guide/factor.md) - 完整的運算子列表
- [因子分析](../user-guide/analyzer.md) - FactorAnalyzer / FactorAnalysisResult 詳細說明
- [策略回測](../user-guide/backtest.md) - VectorizedBacktester 與權重約束
