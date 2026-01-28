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

## 4. 因子分析

```python
from factorium import FactorAnalyzer

# 分析因子表現
analyzer = FactorAnalyzer(
    factor=momentum_rank,
    prices=agg  # 傳入 AggBar，會自動使用 close 欄位
)

# 準備數據（計算未來報酬）
analyzer.prepare_data(periods=[1, 5, 10])

# 計算 IC 分析
ic_summary = analyzer.calculate_ic_summary()
print(ic_summary)

# 計算分層收益
quantile_returns = analyzer.calculate_quantile_returns(quantiles=5)

# 繪製圖表
analyzer.plot_ic(period=1, plot_type='ts')
analyzer.plot_quantile_returns(period=1)
```

---

## 5. 策略回測

```python
from factorium.backtest import Backtester

# 建立回測器
bt = Backtester(
    prices=agg,
    signal=momentum_rank,
    neutralization="market",  # 市場中性
    transaction_cost=0.0003,
    initial_capital=100000
)

# 執行回測
result = bt.run()

# 查看結果
print(result.metrics)
bt.plot_equity()
```

---

## 6. 不同類型的 Bar 聚合

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
- [策略回測](../user-guide/backtest.md) - 回測系統詳細說明
