# Binance 數據載入器 (`factorium/data_loader.py`)

## 概述

`data_loader.py` 提供了一個高階的數據載入器 `BinanceDataLoader`，用於從本地文件讀取 Binance 市場數據。當所需數據不存在時，會自動調用 `BinanceDataDownloader` 下載缺失的數據。

## 類別：`BinanceDataLoader`

### 初始化參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `base_path` | `str` | `"./Data"` | 數據存儲的基礎路徑 |
| `max_concurrent_downloads` | `int` | `5` | 最大並行下載數量 |
| `retry_attempts` | `int` | `3` | 下載失敗時的重試次數 |
| `retry_delay` | `int` | `1` | 重試間隔時間（秒） |

### 主要方法

#### `load_data()`

載入指定參數的市場數據，若本地數據不存在則自動下載。

```python
def load_data(
    self,
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

**參數說明：**

| 參數 | 類型 | 說明 |
|------|------|------|
| `symbol` | `str` | 交易對符號（如 `BTCUSD_PERP`、`BTCUSDT`） |
| `data_type` | `str` | 數據類型：`trades`、`klines`、`aggTrades` |
| `market_type` | `str` | 市場類型：`spot`（現貨）、`futures`（期貨） |
| `futures_type` | `str` | 期貨類型：`cm`（幣本位）、`um`（U本位） |
| `start_date` | `str` | 開始日期，格式 `YYYY-MM-DD` |
| `end_date` | `str` | 結束日期，格式 `YYYY-MM-DD` |
| `days` | `int` | 載入天數（與日期範圍二擇一） |
| `columns` | `List[str]` | 要載入的欄位列表（可選，用於篩選欄位） |
| `force_download` | `bool` | 強制重新下載（即使本地數據存在） |

**回傳值：**

- `pd.DataFrame`：合併後的數據框

## 內部方法

### `_read_data()`

讀取指定日期範圍內的所有數據文件並合併。

### `_read_trades_file()`

讀取交易數據文件（`.csv`）。

- **現貨市場**：CSV 無標頭，欄位為 `id`, `price`, `qty`, `quote_qty`, `time`, `is_buyer_maker`
- **期貨市場**：CSV 有標頭

### `_read_klines_file()`

讀取 K 線數據文件。

### `_read_agg_trades_file()`

讀取聚合交易數據文件。

### `_calculate_date_range()`

計算日期範圍，支援三種模式：

1. 同時指定 `start_date` 和 `end_date`
2. 指定 `start_date` 和 `days`
3. 僅指定 `days`（從當前日期往回推算）

## 數據欄位說明

### Trades（交易數據）

**現貨市場：**

| 欄位 | 說明 |
|------|------|
| `id` | 交易 ID |
| `price` | 成交價格 |
| `qty` | 成交數量 |
| `quote_qty` | 成交金額（報價貨幣） |
| `time` | 成交時間（毫秒時間戳） |
| `is_buyer_maker` | 買方是否為掛單方 |

**期貨市場：**

欄位由 CSV 標頭定義，通常包含類似欄位。

### Klines（K線數據）

K 線數據欄位由 CSV 標頭定義。

### AggTrades（聚合交易）

聚合交易欄位由 CSV 標頭定義。

## 使用範例

### 基本使用

```python
from factorium import BinanceDataLoader

# 創建數據載入器
loader = BinanceDataLoader(base_path="./Data")

# 載入期貨交易數據（使用日期範圍）
df = loader.load_data(
    symbol="BTCUSD_PERP",
    data_type="trades",
    market_type="futures",
    futures_type="cm",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"Loaded {len(df)} records")
print(df.head())
```

### 指定欄位載入

```python
loader = BinanceDataLoader()

# 只載入特定欄位
df = loader.load_data(
    symbol="BTCUSD_PERP",
    data_type="trades",
    market_type="futures",
    futures_type="cm",
    start_date="2024-01-01",
    end_date="2024-01-07",
    columns=["time", "price", "quantity"]
)
```

### 使用天數載入

```python
loader = BinanceDataLoader()

# 載入最近 7 天的數據
df = loader.load_data(
    symbol="BTCUSDT",
    data_type="klines",
    market_type="spot",
    days=7
)
```

### 從指定日期開始載入

```python
loader = BinanceDataLoader()

# 從指定日期開始，載入 10 天
df = loader.load_data(
    symbol="ETHUSDT",
    data_type="aggTrades",
    market_type="spot",
    start_date="2024-06-01",
    days=10
)
```

## 功能特性

1. **自動下載**：當本地數據不存在時，自動調用 `BinanceDataDownloader` 下載
2. **增量檢查**：逐日檢查文件是否存在，只下載缺失的數據
3. **多文件合併**：自動合併多個日期的數據文件為單一 DataFrame
4. **欄位篩選**：支援只載入需要的欄位，減少記憶體使用
5. **格式適配**：自動處理現貨和期貨市場不同的 CSV 格式

## 依賴套件

- `pandas`：數據處理
- `numpy`：數值運算
- `factorium.utils.fetch.BinanceDataDownloader`：數據下載

## 與 `BinanceDataDownloader` 的關係

```
┌─────────────────────────┐
│   BinanceDataLoader     │
│  (factorium/data_loader)│
├─────────────────────────┤
│  - 檢查本地數據         │
│  - 讀取 CSV 文件        │
│  - 合併多日數據         │
│  - 欄位篩選             │
└───────────┬─────────────┘
            │ 調用（數據不存在時）
            ▼
┌─────────────────────────┐
│  BinanceDataDownloader  │
│  (factorium/utils/fetch)│
├─────────────────────────┤
│  - 從 Binance 下載      │
│  - 校驗和驗證           │
│  - 解壓縮               │
└─────────────────────────┘
```

## 注意事項

- 日期範圍的結束日期不包含在載入範圍內（開區間）
- 現貨市場的 trades 數據 CSV 沒有標頭，由程式自動添加欄位名稱
- 若指定的日期範圍內沒有任何數據文件，會拋出 `ValueError`
- 建議在首次使用前確保網路連線正常，以便自動下載數據
