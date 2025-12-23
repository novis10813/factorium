# Binance 數據下載器 (`factorium/utils/fetch.py`)

## 概述

`fetch.py` 是一個用於從 [Binance Vision](https://data.binance.vision/) 下載歷史市場數據的異步下載工具。支援現貨和期貨市場，可下載多種類型的交易數據。

## 類別：`BinanceDataDownloader`

### 初始化參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `base_path` | `str` | `"./Data"` | 數據存儲的根目錄 |
| `max_concurrent_downloads` | `int` | `5` | 最大並行下載數量 |
| `retry_attempts` | `int` | `3` | 下載失敗時的重試次數 |
| `retry_delay` | `int` | `1` | 重試間隔時間（秒） |

### 主要方法

#### `download_data()`

下載指定參數的市場數據。

```python
async def download_data(
    self,
    symbol: str,
    data_type: str,
    market_type: str,
    futures_type: str = 'cm',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: Optional[int] = None
) -> None
```

**參數說明：**

| 參數 | 類型 | 說明 |
|------|------|------|
| `symbol` | `str` | 交易對符號（如 `BTCUSD_PERP`、`BTCUSDT`） |
| `data_type` | `str` | 數據類型：`trades`、`klines`、`aggTrades`、`bookTicker`、`bookDepth` |
| `market_type` | `str` | 市場類型：`spot`（現貨）、`futures`（期貨） |
| `futures_type` | `str` | 期貨類型：`cm`（幣本位）、`um`（U本位），僅期貨市場需要 |
| `start_date` | `str` | 開始日期，格式 `YYYY-MM-DD` |
| `end_date` | `str` | 結束日期，格式 `YYYY-MM-DD` |
| `days` | `int` | 下載天數（與日期範圍二擇一） |

## 支援的數據類型

| 類型 | 說明 |
|------|------|
| `trades` | 逐筆交易數據 |
| `klines` | K線數據（1分鐘） |
| `aggTrades` | 聚合交易數據 |
| `bookTicker` | 最佳買賣報價 |
| `bookDepth` | 訂單簿深度 |

## 數據存儲結構

```
Data/
├── futures/
│   ├── cm/                    # 幣本位期貨
│   │   ├── trades/
│   │   │   └── BTCUSD_PERP/
│   │   ├── klines/
│   │   │   └── BTCUSD_PERP/
│   │   └── aggTrades/
│   │       └── BTCUSD_PERP/
│   └── um/                    # U本位期貨
│       ├── trades/
│       │   └── BTCUSDT/
│       ├── klines/
│       │   └── BTCUSDT/
│       └── aggTrades/
│           └── BTCUSDT/
└── spot/                      # 現貨
    ├── trades/
    │   └── BTCUSDT/
    ├── klines/
    │   └── BTCUSDT/
    └── aggTrades/
        └── BTCUSDT/
```

## 命令列使用

### 基本語法

```bash
python -m factorium.utils.fetch [選項]
```

### 命令列參數

| 參數 | 縮寫 | 預設值 | 說明 |
|------|------|--------|------|
| `--symbol` | `-s` | `BTCUSD_PERP` | 交易對符號 |
| `--data-type` | `-t` | `trades` | 數據類型 |
| `--market-type` | `-m` | `futures` | 市場類型 |
| `--futures-type` | `-f` | `cm` | 期貨類型 |
| `--days` | `-d` | `7` | 下載天數 |
| `--path` | `-p` | `./Data` | 存儲路徑 |
| `--date-range` | `-r` | - | 日期範圍 `YYYY-MM-DD:YYYY-MM-DD` |
| `--max-concurrent` | - | `5` | 最大並行數 |
| `--retry-attempts` | - | `3` | 重試次數 |
| `--retry-delay` | - | `1` | 重試延遲（秒） |

### 使用範例

```bash
# 下載 7 天的幣本位期貨交易數據
python -m factorium.utils.fetch -s BTCUSD_PERP -t trades -m futures -f cm -d 7

# 下載指定日期範圍的 U 本位期貨交易數據
python -m factorium.utils.fetch -s BTCUSDT -t trades -m futures -f um -r 2024-01-01:2024-01-31

# 下載現貨 K 線數據
python -m factorium.utils.fetch -s BTCUSDT -t klines -m spot -r 2024-01-01:2024-01-31

# 下載現貨聚合交易數據，自訂存儲路徑
python -m factorium.utils.fetch -s ETHUSDT -t aggTrades -m spot -d 30 -p ./my_data
```

## 程式內使用

```python
import asyncio
from factorium.utils.fetch import BinanceDataDownloader

async def download_example():
    downloader = BinanceDataDownloader(
        base_path="./Data",
        max_concurrent_downloads=5,
        retry_attempts=3
    )
    
    # 使用日期範圍下載
    await downloader.download_data(
        symbol="BTCUSD_PERP",
        data_type="trades",
        market_type="futures",
        futures_type="cm",
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    # 使用天數下載
    await downloader.download_data(
        symbol="BTCUSDT",
        data_type="klines",
        market_type="spot",
        days=7
    )

if __name__ == "__main__":
    asyncio.run(download_example())
```

## 功能特性

1. **異步並行下載**：使用 `asyncio` 和 `aiohttp` 實現高效並行下載
2. **自動重試機制**：下載失敗時自動重試，可配置重試次數和延遲
3. **校驗和驗證**：自動下載並驗證 SHA256 校驗和，確保數據完整性
4. **自動解壓縮**：下載完成後自動解壓 ZIP 文件並清理壓縮檔
5. **自動更新 README**：每次下載完成後自動更新數據目錄的 README 文件

## 依賴套件

- `aiohttp`：異步 HTTP 請求
- `aiofiles`：異步文件操作
- `asyncio`：異步程式框架

## 注意事項

- 日期範圍的結束日期不包含在下載範圍內（開區間）
- 使用 `--date-range` 時會覆蓋 `--days` 參數
- 幣本位期貨（cm）使用 `USD` 計價符號（如 `BTCUSD_PERP`）
- U本位期貨（um）和現貨使用 `USDT` 計價符號（如 `BTCUSDT`）
