# Factorium

量化因子分析與研究工具庫。

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 目錄

- [專案說明](#專案說明)
- [安裝](#安裝)
- [快速開始](#快速開始)
- [核心元件](#核心元件)
  - [BinanceDataLoader - 資料載入器](#binancedataloader---資料載入器)
  - [Bar - K棒取樣](#bar---k棒取樣)
  - [AggBar - 多標的資料容器](#aggbar---多標的資料容器)
  - [Factor - 因子運算](#factor---因子運算)
- [因子運算子](#因子運算子)
  - [時間序列運算子](#時間序列運算子-time-series-operations)
  - [橫截面運算子](#橫截面運算子-cross-sectional-operations)
  - [數學運算子](#數學運算子-math-operations)
- [完整範例](#完整範例)
- [測試](#測試)

## 專案說明

Factorium 是一個專為量化金融研究設計的 Python 工具庫，提供：

- 🔄 **多種 K 棒取樣方法**：時間棒、Tick 棒、成交量棒、金額棒
- 📊 **豐富的因子運算子**：時間序列、橫截面、數學運算
- 📥 **Binance 歷史資料下載**：自動從 Binance Vision 下載資料
- ⚡ **高效能運算**：使用 Numba JIT 加速關鍵運算

## 安裝

```bash
# 使用 uv (推薦)
uv add factorium

# 或使用 pip
pip install factorium
```

### 開發環境安裝

```bash
git clone https://github.com/novis10813/factorium.git
cd factorium
uv sync --dev
```

## 快速開始

```python
from factorium import BinanceDataLoader, TimeBar, AggBar

# 1. 載入資料
loader = BinanceDataLoader()
df = loader.load_data(
    symbol="BTCUSDT",
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=7
)

# 2. 建立 K 棒 (1 分鐘)
bar = TimeBar(df, timestamp_col="transact_time", price_col="price", 
              volume_col="quantity", interval_ms=60_000)

# 3. 建立多標的容器
agg = AggBar([bar])

# 4. 提取因子並進行運算
close = agg['close']
momentum = close.ts_delta(20) / close.ts_shift(20)
ranked = momentum.rank()
```

## 核心元件

### BinanceDataLoader - 資料載入器

`BinanceDataLoader` 提供同步介面，從 Binance Vision 載入歷史市場資料。若本地檔案不存在，會自動下載。

#### 基本用法

```python
from factorium import BinanceDataLoader

loader = BinanceDataLoader(
    base_path="./Data",              # 資料儲存路徑
    max_concurrent_downloads=5,       # 最大併發下載數
    retry_attempts=3,                 # 下載失敗重試次數
    retry_delay=1                     # 重試間隔（秒）
)
```

#### 載入資料

```python
# 載入期貨交易資料 (USDT 本位)
df = loader.load_data(
    symbol="BTCUSDT",
    data_type="aggTrades",      # trades / klines / aggTrades
    market_type="futures",      # spot / futures
    futures_type="um",          # um (USDT本位) / cm (幣本位)
    start_date="2024-01-01",
    end_date="2024-01-07"
)

# 或使用天數
df = loader.load_data(
    symbol="ETHUSDT",
    data_type="trades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=30
)

# 載入現貨資料
df = loader.load_data(
    symbol="BTCUSDT",
    data_type="klines",
    market_type="spot",
    start_date="2024-01-01",
    days=7
)

# 強制重新下載
df = loader.load_data(
    symbol="BTCUSDT",
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=7,
    force_download=True
)

# 只讀取特定欄位
df = loader.load_data(
    symbol="BTCUSDT",
    data_type="trades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=7,
    columns=["time", "price", "qty"]
)
```

#### 資料類型說明

| data_type | 說明 | 常用欄位 |
|-----------|------|----------|
| `trades` | 逐筆成交 | id, price, qty, time, is_buyer_maker |
| `aggTrades` | 聚合成交 | agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker |
| `klines` | K 線資料 | open_time, open, high, low, close, volume, ... |

#### 命令列工具

也可以直接從命令列下載資料：

```bash
# 下載 7 天的期貨交易資料 (幣本位)
python -m factorium.utils.fetch -s BTCUSD_PERP -t trades -m futures -f cm -d 7

# 下載指定日期範圍 (USDT 本位)
python -m factorium.utils.fetch -s BTCUSDT -t aggTrades -m futures -f um -r 2024-01-01:2024-01-31

# 下載現貨 K 線資料
python -m factorium.utils.fetch -s BTCUSDT -t klines -m spot -r 2024-01-01:2024-01-31
```

---

### Bar - K棒取樣

Factorium 提供四種 K 棒取樣方法，將 tick 級別資料聚合成 OHLCV 格式：

| 類別 | 說明 | 適用場景 |
|------|------|----------|
| `TimeBar` | 固定時間間隔 | 一般技術分析 |
| `TickBar` | 固定 tick 數量 | 交易活躍度分析 |
| `VolumeBar` | 固定成交量 | 流動性分析 |
| `DollarBar` | 固定成交金額 | 資金流向分析 |

#### TimeBar - 時間棒

以固定時間間隔聚合資料，最常見的 K 棒類型。

```python
from factorium import TimeBar

# 建立 1 分鐘 K 棒
bar = TimeBar(
    df,
    timestamp_col='transact_time',  # 時間戳欄位（毫秒）
    price_col='price',              # 價格欄位
    volume_col='quantity',          # 成交量欄位
    interval_ms=60_000              # 間隔（毫秒），60000 = 1分鐘
)

# 建立 5 分鐘 K 棒
bar_5m = TimeBar(df, timestamp_col='transact_time', price_col='price', 
                 volume_col='quantity', interval_ms=300_000)

# 建立 1 小時 K 棒
bar_1h = TimeBar(df, timestamp_col='transact_time', price_col='price', 
                 volume_col='quantity', interval_ms=3_600_000)

# 存取聚合後的資料
print(bar.bars)
# 輸出欄位：symbol, start_time, end_time, open, high, low, close, volume
#          (若有 is_buyer_maker 欄位) num_buyer, num_seller, num_buyer_volume, num_seller_volume
```

#### TickBar - Tick 棒

每固定數量的 tick（成交筆數）形成一根 K 棒。

```python
from factorium import TickBar

# 每 1000 筆成交形成一根 K 棒
bar = TickBar(
    df,
    timestamp_col='ts_init',
    price_col='price',
    volume_col='size',
    interval_ticks=1000
)

print(len(bar))  # K 棒數量
```

#### VolumeBar - 成交量棒

每累積固定成交量形成一根 K 棒。

```python
from factorium import VolumeBar

# 每累積 100 BTC 成交量形成一根 K 棒
bar = VolumeBar(
    df,
    timestamp_col='time',
    price_col='price',
    volume_col='qty',
    interval_volume=100
)
```

#### DollarBar - 金額棒

每累積固定成交金額形成一根 K 棒。

```python
from factorium import DollarBar

# 每累積 1,000,000 USD 形成一根 K 棒
bar = DollarBar(
    df,
    timestamp_col='ts_init',
    price_col='price',
    volume_col='size',
    interval_dollar=1_000_000
)
```

#### 使用 apply 添加自訂特徵

所有 Bar 類別都支援 `apply` 方法來添加自訂欄位：

```python
bar = TimeBar(df, interval_ms=60_000)

# 添加技術指標
bar.apply({
    'sma_20': lambda bars: bars['close'].rolling(20).mean(),
    'forward_return_5': lambda bars: (bars['close'].shift(-5) - bars['close']) / bars['close'],
    'volatility': lambda bars: bars['close'].rolling(20).std(),
})

print(bar.bars.columns)
# ['symbol', 'start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume', 
#  'sma_20', 'forward_return_5', 'volatility']
```

---

### AggBar - 多標的資料容器

`AggBar` 用於管理多個標的的 K 棒資料，提供統一的介面進行因子提取和資料操作。

#### 建立 AggBar

```python
from factorium import AggBar, TimeBar

# 方法 1：從多個 Bar 物件建立
bar_btc = TimeBar(df_btc, interval_ms=60_000)
bar_eth = TimeBar(df_eth, interval_ms=60_000)
agg = AggBar([bar_btc, bar_eth])

# 方法 2：從 DataFrame 建立（需包含 start_time, end_time, symbol 欄位）
agg = AggBar.from_df(df)

# 方法 3：從 CSV 檔案建立
agg = AggBar.from_csv("./data/aggregated.csv")
```

#### 基本操作

```python
# 查看基本資訊
print(agg)
# AggBar: 10000 rows, 8 columns, symbols=2, time_range=2024-01-01 00:00:00 - 2024-01-07 23:59:00

# 取得欄位列表
print(agg.cols)
# ['start_time', 'end_time', 'symbol', 'open', 'high', 'low', 'close', 'volume']

# 取得標的列表
print(agg.symbols)
# ['BTCUSDT', 'ETHUSDT']

# 取得時間戳記
print(agg.timestamps)

# 取得各標的摘要資訊
print(agg.info())
#          num_kbar          start_time            end_time  num_nan
# BTCUSDT     5000  2024-01-01 00:00:00  2024-01-07 23:59:00        0
# ETHUSDT     5000  2024-01-01 00:00:00  2024-01-07 23:59:00        0
```

#### 提取因子

使用 `[]` 運算子從 AggBar 提取欄位作為 Factor：

```python
# 提取單一欄位 -> 返回 Factor
close = agg['close']
volume = agg['volume']

# 提取多個欄位 -> 返回新的 AggBar
ohlc = agg[['open', 'high', 'low', 'close']]
```

#### 資料切片

使用 `slice` 方法按時間和標的篩選資料：

```python
# 按時間範圍篩選
sliced = agg.slice(
    start="2024-01-02 00:00:00",
    end="2024-01-05 23:59:59"
)

# 按標的篩選
btc_only = agg.slice(symbols=["BTCUSDT"])

# 同時篩選時間和標的
filtered = agg.slice(
    start="2024-01-02",
    end="2024-01-05",
    symbols=["BTCUSDT", "ETHUSDT"]
)

# 也可以使用 timestamp（毫秒或秒）
sliced = agg.slice(start=1704153600000, end=1704499200000)
```

#### 儲存資料

```python
# 儲存為 CSV
agg.to_csv("./output/data.csv")

# 儲存為 Parquet（推薦，檔案較小且讀取較快）
agg.to_parquet("./output/data.parquet")

# 轉換為 DataFrame
df = agg.to_df()
```

---

### Factor - 因子運算

`Factor` 是 Factorium 的核心類別，代表一個多標的時間序列因子，支援豐富的運算子。

#### 建立 Factor

```python
from factorium import Factor, AggBar

# 從 AggBar 提取
agg = AggBar([bar1, bar2])
close = agg['close']

# 從 DataFrame 建立
factor = Factor(df, name="my_factor")

# 從檔案建立
factor = Factor("./factors/momentum.csv", name="momentum")
factor = Factor("./factors/momentum.parquet", name="momentum")
```

#### 基本運算

```python
close = agg['close']
volume = agg['volume']

# 算術運算
returns = (close - close.ts_shift(1)) / close.ts_shift(1)
vwap = (close * volume) / volume

# 比較運算（返回 0/1）
is_up = close > close.ts_shift(1)
is_high_volume = volume > volume.ts_mean(20)
```

---

## 因子運算子

### 時間序列運算子 (Time-Series Operations)

對每個標的分別計算滾動窗口統計量：

| 方法 | 說明 | 範例 |
|------|------|------|
| `ts_mean(window)` | 滾動平均 | `close.ts_mean(20)` |
| `ts_std(window)` | 滾動標準差 | `close.ts_std(20)` |
| `ts_sum(window)` | 滾動加總 | `volume.ts_sum(10)` |
| `ts_product(window)` | 滾動乘積 | `returns.ts_product(5)` |
| `ts_min(window)` | 滾動最小值 | `low.ts_min(20)` |
| `ts_max(window)` | 滾動最大值 | `high.ts_max(20)` |
| `ts_median(window)` | 滾動中位數 | `close.ts_median(20)` |
| `ts_rank(window)` | 時間序列排名（百分位） | `close.ts_rank(20)` |
| `ts_argmin(window)` | 最小值距今期數 | `close.ts_argmin(20)` |
| `ts_argmax(window)` | 最大值距今期數 | `close.ts_argmax(20)` |
| `ts_shift(period)` | 延遲（lag） | `close.ts_shift(1)` |
| `ts_delta(period)` | 差分 | `close.ts_delta(1)` |
| `ts_zscore(window)` | Z-score 標準化 | `close.ts_zscore(20)` |
| `ts_scale(window)` | Min-Max 標準化 | `close.ts_scale(20)` |
| `ts_quantile(window, driver)` | 分位數轉換 | `close.ts_quantile(20, "gaussian")` |
| `ts_skewness(window)` | 滾動偏度 | `returns.ts_skewness(20)` |
| `ts_kurtosis(window)` | 滾動峰度 | `returns.ts_kurtosis(20)` |
| `ts_corr(other, window)` | 滾動相關係數 | `close.ts_corr(volume, 20)` |
| `ts_cov(other, window)` | 滾動共變異數 | `close.ts_cov(volume, 20)` |
| `ts_cv(window)` | 變異係數 | `close.ts_cv(20)` |
| `ts_autocorr(window, lag)` | 自相關係數 | `returns.ts_autocorr(20, 1)` |
| `ts_jumpiness(window)` | 跳躍性指標 | `close.ts_jumpiness(20)` |
| `ts_reversal_count(window)` | 反轉次數 | `close.ts_reversal_count(20)` |
| `ts_vr(window, k)` | 變異數比率 | `close.ts_vr(20, 2)` |
| `ts_step(start)` | 累計期數 | `close.ts_step(1)` |

#### 使用範例

```python
close = agg['close']
volume = agg['volume']

# 動量因子
momentum = close.ts_delta(20) / close.ts_shift(20)

# 波動率因子
volatility = close.ts_std(20) / close.ts_mean(20)

# 成交量異常
volume_zscore = volume.ts_zscore(20)

# 價量相關性
price_volume_corr = close.ts_corr(volume, 20)

# 變異數比率（測試隨機漫步假說）
# VR ≈ 1: 隨機漫步, VR > 1: 趨勢, VR < 1: 均值回歸
vr = close.ts_vr(20, 2)
```

---

### 橫截面運算子 (Cross-Sectional Operations)

對同一時間點的所有標的計算統計量：

| 方法 | 說明 | 範例 |
|------|------|------|
| `rank()` | 橫截面百分位排名 | `momentum.rank()` |
| `mean()` | 橫截面平均 | `returns.mean()` |
| `median()` | 橫截面中位數 | `returns.median()` |

#### 使用範例

```python
close = agg['close']

# 計算動量並排名
momentum = close.ts_delta(20) / close.ts_shift(20)
momentum_rank = momentum.rank()  # 每個時間點，對所有標的排名

# 市場調整報酬
returns = close.ts_delta(1) / close.ts_shift(1)
market_return = returns.mean()  # 每個時間點的市場平均報酬
excess_return = returns - market_return  # 超額報酬
```

---

### 數學運算子 (Math Operations)

基本數學函數：

| 方法 | 說明 | 範例 |
|------|------|------|
| `abs()` | 絕對值 | `returns.abs()` |
| `sign()` | 符號函數 | `returns.sign()` |
| `log(base)` | 對數（預設自然對數） | `close.log()` |
| `ln()` | 自然對數 | `close.ln()` |
| `sqrt()` | 平方根 | `variance.sqrt()` |
| `pow(exp)` | 次方 | `returns.pow(2)` |
| `signed_pow(exp)` | 保留符號的次方 | `returns.signed_pow(0.5)` |
| `signed_log1p()` | 保留符號的 log(1+x) | `returns.signed_log1p()` |
| `inverse()` | 倒數 | `close.inverse()` |
| `max(other)` | 逐元素取最大 | `close.max(100)` |
| `min(other)` | 逐元素取最小 | `close.min(100)` |
| `where(cond, other)` | 條件選擇 | `close.where(is_valid, 0)` |
| `reverse()` | 取負值 | `momentum.reverse()` |

#### 使用範例

```python
close = agg['close']
returns = close.ts_delta(1) / close.ts_shift(1)

# 對數報酬
log_returns = close.log().ts_delta(1)

# 處理極端值
capped_returns = returns.max(-0.1).min(0.1)

# 條件因子
is_positive = returns > 0
positive_returns = returns.where(is_positive, 0)

# 保留符號的平方根（用於處理負值）
signed_sqrt = returns.signed_pow(0.5)
```

---

## 完整範例

### 範例 1：動量因子研究

```python
from factorium import BinanceDataLoader, TimeBar, AggBar

# 1. 載入多個標的資料
loader = BinanceDataLoader()
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
bars = []

for symbol in symbols:
    df = loader.load_data(
        symbol=symbol,
        data_type="aggTrades",
        market_type="futures",
        futures_type="um",
        start_date="2024-01-01",
        days=30
    )
    bar = TimeBar(df, timestamp_col="transact_time", price_col="price", 
                  volume_col="quantity", interval_ms=60_000)
    bars.append(bar)

# 2. 建立 AggBar
agg = AggBar(bars)
print(agg.info())

# 3. 計算因子
close = agg['close']
volume = agg['volume']

# 價格動量
momentum_5 = close.ts_delta(5) / close.ts_shift(5)
momentum_20 = close.ts_delta(20) / close.ts_shift(20)

# 成交量加權動量
vwap = (close * volume).ts_sum(20) / volume.ts_sum(20)
vwap_deviation = (close - vwap) / vwap

# 波動調整動量
volatility = close.ts_std(20)
risk_adjusted_momentum = momentum_20 / volatility

# 4. 橫截面排名
momentum_rank = risk_adjusted_momentum.rank()

# 5. 輸出結果
print(momentum_rank.data.tail(20))
```

### 範例 2：均值回歸因子

```python
from factorium import BinanceDataLoader, TimeBar, AggBar

loader = BinanceDataLoader()
df = loader.load_data(
    symbol="BTCUSDT",
    data_type="aggTrades", 
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=30
)

bar = TimeBar(df, timestamp_col="transact_time", price_col="price",
              volume_col="quantity", interval_ms=60_000)
agg = AggBar([bar])

close = agg['close']

# Z-score 均值回歸
zscore = close.ts_zscore(20)

# Bollinger Band 位置
sma = close.ts_mean(20)
std = close.ts_std(20)
bb_position = (close - sma) / (2 * std)

# RSI-like 指標
delta = close.ts_delta(1)
gain = delta.where(delta > 0, 0)
loss = (-delta).where(delta < 0, 0)
avg_gain = gain.ts_mean(14)
avg_loss = loss.ts_mean(14)
rs = avg_gain / avg_loss
rsi = 1 - (1 / (1 + rs))

print(rsi.data.tail(20))
```

### 範例 3：使用不同 Bar 類型

```python
from factorium import BinanceDataLoader, TimeBar, TickBar, VolumeBar, DollarBar, AggBar

loader = BinanceDataLoader()
df = loader.load_data(
    symbol="BTCUSDT",
    data_type="trades",
    market_type="futures", 
    futures_type="um",
    start_date="2024-01-01",
    days=1
)

# 比較不同取樣方法
time_bar = TimeBar(df, timestamp_col="time", price_col="price", 
                   volume_col="qty", interval_ms=60_000)
tick_bar = TickBar(df, timestamp_col="time", price_col="price",
                   volume_col="qty", interval_ticks=1000)
volume_bar = VolumeBar(df, timestamp_col="time", price_col="price",
                       volume_col="qty", interval_volume=100)
dollar_bar = DollarBar(df, timestamp_col="time", price_col="price",
                       volume_col="qty", interval_dollar=1_000_000)

print(f"TimeBar: {len(time_bar)} bars")
print(f"TickBar: {len(tick_bar)} bars")
print(f"VolumeBar: {len(volume_bar)} bars")
print(f"DollarBar: {len(dollar_bar)} bars")
```

---

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

執行並顯示覆蓋率：

```bash
uv run pytest --cov=factorium
```

更多關於測試策略的細節，特別是數學運算子的「雙向驗證」流程，請參閱 [docs/pytest.md](docs/pytest.md)。

---

## 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案。

## 作者

Samuel Chang
