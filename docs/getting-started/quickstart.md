# 五分鐘教學

本教學將帶你快速上手 Factorium 的核心功能。

---

## 1. 載入數據

```python
from factorium import BinanceDataLoader, AggBar

# 下載並載入 BTC 期貨交易數據
loader = BinanceDataLoader()
df = loader.load_data(
    symbol="BTCUSD_PERP",
    data_type="trades",
    market_type="futures",
    futures_type="cm",
    days=7
)

print(f"載入 {len(df):,} 筆交易")
```

---

## 2. 建立 K 線（Bar 聚合）

```python
from factorium import TimeBar

# 將 tick 數據聚合成 1 小時 K 線
bar = TimeBar(trades=df, resolution=60)  # 60 分鐘
ohlcv = bar.get_bars()

print(ohlcv.head())
```

---

## 3. 建立多標的容器

```python
# 假設有多個標的的 K 線數據
agg = AggBar.from_parquet("data/crypto_1h.parquet")

# 查看結構
print(f"標的數量: {agg.n_symbols}")
print(f"時間點數量: {agg.n_timestamps}")
print(f"欄位: {agg.cols}")
```

---

## 4. 計算因子

```python
# 動量因子：過去 20 期報酬
momentum = agg["close"] / agg["close"].ts_shift(20) - 1

# 波動率因子：過去 20 期標準差
volatility = agg["close"].ts_pct_change().ts_std(20)

# 橫截面排名
momentum_rank = momentum.cs_rank()

print(momentum_rank.data.head())
```

---

## 5. 因子分析

```python
from factorium import FactorAnalyzer

# 計算未來報酬
forward_returns = agg["close"].ts_pct_change().ts_shift(-1)

# 分析因子表現
analyzer = FactorAnalyzer(
    factor=momentum_rank,
    forward_returns=forward_returns,
    n_quantiles=5
)

print(analyzer.summary())
analyzer.plot_quantile_returns()
```

---

## 6. 策略回測

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

## 下一步

- [Bar 聚合](../user-guide/bar.md) - 深入了解不同類型的 K 線
- [Factor 因子](../user-guide/factor.md) - 完整的運算子列表
- [策略回測](../user-guide/backtest.md) - 回測系統詳細說明
