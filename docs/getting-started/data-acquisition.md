# 資料獲取

本頁說明如何下載和載入 Binance 市場數據。

---

## 快速開始

```python
from factorium import BinanceDataLoader

loader = BinanceDataLoader(base_path="./Data")

# 載入最近 7 天的交易數據並聚合成 1 分鐘 K 棒
agg = loader.load_aggbar(
    symbols=["BTCUSDT"],
    data_type="aggTrades",
    market_type="futures",
    futures_type="um",
    start_date="2024-01-01",
    days=7,
    bar_type="time",
    interval=60_000,  # 1 分鐘
)

# 提取因子
close = agg['close']
momentum = close.ts_delta(20) / close.ts_shift(20)
```

---

## 數據載入器

`BinanceDataLoader` 提供高階的數據載入介面，當本地數據不存在時會自動下載。

### 初始化

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `base_path` | `str` | `"./Data"` | 數據存儲的根目錄 |
| `max_concurrent_downloads` | `int` | `5` | 最大並行下載數量 |
| `retry_attempts` | `int` | `3` | 下載失敗時的重試次數 |

### load_aggbar() 方法

```python
def load_aggbar(
    symbols: List[str],
    data_type: str,
    market_type: str,
    futures_type: str = 'um',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
    bar_type: Literal["time", "tick", "volume", "dollar"] = "time",
    interval: int = 60_000,
    force_download: bool = False,
    use_cache: bool = True
) -> AggBar
```

| 參數 | 類型 | 說明 |
|------|------|------|
| `symbols` | `List[str]` | 交易對符號列表（如 `["BTCUSDT", "ETHUSDT"]`） |
| `data_type` | `str` | 數據類型：`trades`、`klines`、`aggTrades` |
| `market_type` | `str` | 市場類型：`spot`（現貨）、`futures`（期貨） |
| `futures_type` | `str` | 期貨類型：`cm`（幣本位）、`um`（U本位） |
| `start_date` | `str` | 開始日期，格式 `YYYY-MM-DD` |
| `end_date` | `str` | 結束日期，格式 `YYYY-MM-DD` |
| `days` | `int` | 載入天數（與日期範圍二擇一） |
| `bar_type` | `str` | K 棒類型：`time`、`tick`、`volume`、`dollar` |
| `interval` | `int` | K 棒間隔（時間棒為毫秒，其他為對應單位） |
| `force_download` | `bool` | 強制重新下載 |
| `use_cache` | `bool` | 使用快取（避免重複聚合） |

### Bar 類型說明

| bar_type | interval 參數 | 說明 | 多標的支援 |
|----------|--------------|------|-----------|
| `time` | 毫秒 | 固定時間間隔（如 60_000 = 1分鐘） | ✓ |
| `tick` | tick 數量 | 固定 tick 數量形成一根 K 棒 | ✗ |
| `volume` | 成交量 | 固定成交量形成一根 K 棒 | ✗ |
| `dollar` | 金額 | 固定成交金額形成一根 K 棒 | ✗ |

### 範例

=== "單標的時間棒"

    ```python
    agg = loader.load_aggbar(
        symbols=["BTCUSDT"],
        data_type="aggTrades",
        market_type="futures",
        futures_type="um",
        start_date="2024-01-01",
        days=7,
        bar_type="time",
        interval=60_000,  # 1 分鐘
    )
    ```

=== "多標的時間棒"

    ```python
    agg = loader.load_aggbar(
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        data_type="aggTrades",
        market_type="futures",
        futures_type="um",
        start_date="2024-01-01",
        days=7,
        bar_type="time",
        interval=300_000,  # 5 分鐘
    )
    ```

=== "Tick 棒"

    ```python
    agg = loader.load_aggbar(
        symbols=["BTCUSDT"],  # 僅支援單一標的
        data_type="trades",
        market_type="futures",
        futures_type="um",
        start_date="2024-01-01",
        days=1,
        bar_type="tick",
        interval=1000,  # 每 1000 筆成交
    )
    ```

=== "成交量棒"

    ```python
    agg = loader.load_aggbar(
        symbols=["BTCUSDT"],  # 僅支援單一標的
        data_type="trades",
        market_type="futures",
        futures_type="um",
        start_date="2024-01-01",
        days=1,
        bar_type="volume",
        interval=100,  # 每 100 BTC
    )
    ```

=== "金額棒"

    ```python
    agg = loader.load_aggbar(
        symbols=["BTCUSDT"],  # 僅支援單一標的
        data_type="trades",
        market_type="futures",
        futures_type="um",
        start_date="2024-01-01",
        days=1,
        bar_type="dollar",
        interval=1_000_000,  # 每 100 萬 USD
    )
    ```

=== "載入 Klines 數據"

    Klines 數據是來自 Binance 的已聚合 OHLCV 數據。與 trades/aggTrades 不同，klines 不需要進行棒聚合，直接載入。

    ```python
    from factorium.data import BinanceDataLoader

    loader = BinanceDataLoader()

    # 載入 1m klines（預設）
    agg = loader.load_aggbar(
        symbols=["BTCUSDT", "ETHUSDT"],
        data_type="klines",
        market_type="futures",
        futures_type="um",
        start_date="2024-01-01",
        days=7,
    )

    # 重新取樣為 5m
    agg_5m = loader.load_aggbar(
        symbols=["BTCUSDT"],
        data_type="klines",
        market_type="futures",
        futures_type="um",
        start_date="2024-01-01",
        days=7,
        interval=300_000,  # 5 分鐘
    )

    # 重新取樣為 1h
    agg_1h = loader.load_aggbar(
        symbols=["BTCUSDT"],
        data_type="klines",
        market_type="futures",
        futures_type="um",
        start_date="2024-01-01",
        days=7,
        interval=3_600_000,  # 1 小時
    )
    ```

    **注意事項：**
    - Klines 僅支援 `bar_type="time"`（預設值）
    - 下載的數據始終為 1 分鐘，重新取樣即時進行
    - Klines 繞過 `BarAggregator` 以提升效能
    - 載入所有 klines 欄位，包括微觀結構數據（quote_volume, count, taker_buy_volume, taker_buy_quote_volume）

---

## Jupyter Notebook 支援

在 Jupyter Notebook 中使用時，建議安裝 `nest-asyncio` 以獲得最佳體驗：

```bash
pip install factorium[jupyter]
# 或
pip install nest-asyncio
```

---

## 命令列下載

直接使用命令列下載數據：

```bash
# 下載 7 天的幣本位期貨交易數據
python -m factorium.utils.fetch -s BTCUSD_PERP -t trades -m futures -f cm -d 7

# 下載指定日期範圍的 U 本位期貨
python -m factorium.utils.fetch -s BTCUSDT -t trades -m futures -f um -r 2024-01-01:2024-01-31

# 下載現貨 K 線數據
python -m factorium.utils.fetch -s BTCUSDT -t klines -m spot -r 2024-01-01:2024-01-31
```

### CLI 參數

| 參數 | 縮寫 | 預設值 | 說明 |
|------|------|--------|------|
| `--symbol` | `-s` | `BTCUSD_PERP` | 交易對符號 |
| `--data-type` | `-t` | `trades` | 數據類型 |
| `--market-type` | `-m` | `futures` | 市場類型 |
| `--futures-type` | `-f` | `cm` | 期貨類型 |
| `--days` | `-d` | `7` | 下載天數 |
| `--path` | `-p` | `./Data` | 存儲路徑 |
| `--date-range` | `-r` | - | 日期範圍 `YYYY-MM-DD:YYYY-MM-DD` |

---

## 支援的數據類型

| 類型 | 說明 |
|------|------|
| `trades` | 逐筆交易數據 |
| `klines` | K 線數據（1 分鐘） |
| `aggTrades` | 聚合交易數據 |
| `bookTicker` | 最佳買賣報價 |
| `bookDepth` | 訂單簿深度 |

---

## 數據存儲結構

數據使用 Hive 分區格式存儲：

```
Data/
├── market=futures_um/
│   └── data_type=aggTrades/
│       └── symbol=BTCUSDT/
│           ├── year=2024/
│           │   └── month=01/
│           │       ├── day=01/
│           │       │   └── data.parquet
│           │       └── day=02/
│           │           └── data.parquet
│           └── ...
└── market=spot/
    └── data_type=trades/
        └── symbol=BTCUSDT/
            └── ...
```

---

## 注意事項

!!! warning "日期區間"
    結束日期不包含在載入範圍內（開區間）

!!! info "符號命名"
    - 幣本位期貨（cm）使用 `USD` 計價：`BTCUSD_PERP`
    - U 本位期貨（um）和現貨使用 `USDT` 計價：`BTCUSDT`

!!! tip "快取機制"
    `load_aggbar` 內建快取機制，第二次載入相同參數時會直接從快取讀取，大幅提升載入速度。使用 `use_cache=False` 可停用快取。
