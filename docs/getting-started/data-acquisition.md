# 資料獲取

本頁說明如何下載和載入 Binance 市場數據。

---

## 快速開始

```python
from factorium import BinanceDataLoader

loader = BinanceDataLoader(base_path="./Data")

# 載入最近 7 天的交易數據（自動下載缺失數據）
df = loader.load_data(
    symbol="BTCUSD_PERP",
    data_type="trades",
    market_type="futures",
    futures_type="cm",
    days=7
)
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

### load_data() 方法

```python
def load_data(
    symbol: str,
    data_type: str,
    market_type: str,
    futures_type: str = 'cm',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None,
    columns: Optional[List[str]] = None,
    force_download: bool = False
) -> pd.DataFrame
```

| 參數 | 類型 | 說明 |
|------|------|------|
| `symbol` | `str` | 交易對符號（如 `BTCUSD_PERP`、`BTCUSDT`） |
| `data_type` | `str` | 數據類型：`trades`、`klines`、`aggTrades` |
| `market_type` | `str` | 市場類型：`spot`（現貨）、`futures`（期貨） |
| `futures_type` | `str` | 期貨類型：`cm`（幣本位）、`um`（U本位） |
| `start_date` | `str` | 開始日期，格式 `YYYY-MM-DD` |
| `end_date` | `str` | 結束日期，格式 `YYYY-MM-DD` |
| `days` | `int` | 載入天數（與日期範圍二擇一） |
| `columns` | `List[str]` | 要載入的欄位列表 |
| `force_download` | `bool` | 強制重新下載 |

### 範例

=== "日期範圍"

    ```python
    df = loader.load_data(
        symbol="BTCUSD_PERP",
        data_type="trades",
        market_type="futures",
        futures_type="cm",
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    ```

=== "天數"

    ```python
    df = loader.load_data(
        symbol="BTCUSDT",
        data_type="klines",
        market_type="spot",
        days=7
    )
    ```

=== "篩選欄位"

    ```python
    df = loader.load_data(
        symbol="BTCUSD_PERP",
        data_type="trades",
        market_type="futures",
        columns=["time", "price", "quantity"]
    )
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

```
Data/
├── futures/
│   ├── cm/                    # 幣本位期貨
│   │   ├── trades/
│   │   │   └── BTCUSD_PERP/
│   │   └── klines/
│   │       └── BTCUSD_PERP/
│   └── um/                    # U 本位期貨
│       └── trades/
│           └── BTCUSDT/
└── spot/                      # 現貨
    ├── trades/
    │   └── BTCUSDT/
    └── klines/
        └── BTCUSDT/
```

---

## 注意事項

!!! warning "日期區間"
    結束日期不包含在載入範圍內（開區間）

!!! info "符號命名"
    - 幣本位期貨（cm）使用 `USD` 計價：`BTCUSD_PERP`
    - U 本位期貨（um）和現貨使用 `USDT` 計價：`BTCUSDT`
