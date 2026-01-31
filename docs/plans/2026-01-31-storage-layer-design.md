# Storage Layer 抽象化設計

> 日期：2026-01-31
> 狀態：設計完成，待實作

## 概述

在 Data 模組中加入 Storage Layer 抽象層，讓用戶可以選擇將資料儲存在本地端或雲端（如 S3）。

## 設計決策

| 項目 | 決策 |
|------|------|
| 抽象範圍 | Cache + 原始資料都抽象化 |
| 首批後端 | Local + S3 |
| 認證方式 | 環境變數（AWS 標準） |
| API 風格 | 分離參數：`backend='s3'`, `path='bucket/path'` |
| 依賴管理 | Optional dependency (`factorium[s3]`) |

## 分支策略

此功能應在 `dev` 分支上開發，原因：

1. 這是新功能，不是緊急 bug 修復
2. 改動範圍較大，影響多個模組
3. 不影響現有用戶，符合 minor version 發布策略

## 架構設計

### 目錄結構

```
src/factorium/
├── storage/                    # 新增的 Storage 模組
│   ├── __init__.py
│   ├── base.py                 # StorageBackend 抽象基類
│   ├── local.py                # LocalStorageBackend
│   └── s3.py                   # S3StorageBackend
└── data/
    ├── cache.py                # 修改：注入 StorageBackend
    └── loader.py               # 修改：注入 StorageBackend
```

### StorageBackend 抽象介面

```python
# storage/base.py
from abc import ABC, abstractmethod
from typing import List
import polars as pl

class StorageBackend(ABC):
    """儲存後端抽象基類"""
    
    @abstractmethod
    def read_parquet(self, path: str) -> pl.DataFrame:
        """讀取單一 Parquet 檔案"""
        ...
    
    @abstractmethod
    def write_parquet(self, df: pl.DataFrame, path: str) -> None:
        """寫入 Parquet 檔案"""
        ...
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """檢查路徑是否存在"""
        ...
    
    @abstractmethod
    def glob(self, pattern: str) -> List[str]:
        """列出符合 pattern 的檔案"""
        ...
    
    @abstractmethod
    def delete(self, path: str) -> None:
        """刪除檔案"""
        ...
    
    @abstractmethod
    def makedirs(self, path: str) -> None:
        """建立目錄結構"""
        ...
```

### Factory 函數

```python
def get_storage_backend(backend: str = "local", path: str = "./Data") -> StorageBackend:
    if backend == "local":
        return LocalStorageBackend(path)
    elif backend == "s3":
        return S3StorageBackend(
            bucket=path.split("/")[0], 
            prefix="/".join(path.split("/")[1:])
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
```

## 使用方式

```python
# 本地端（預設，向後相容）
loader = BinanceDataLoader(backend="local", path="./Data")

# S3
loader = BinanceDataLoader(backend="s3", path="my-bucket/data")
```

## 對現有元件的改動

### BarCache

```python
# 之前
class BarCache:
    def __init__(self, cache_dir: Path = Path("./Data/.cache")):
        self.cache_dir = Path(cache_dir)

# 之後
class BarCache:
    def __init__(self, storage: StorageBackend, cache_prefix: str = ".cache"):
        self.storage = storage
        self.cache_prefix = cache_prefix
```

### BinanceDataLoader

```python
# 之前
class BinanceDataLoader:
    def __init__(self, base_path: str = "./Data", ...):
        self.base_path = Path(base_path)

# 之後
class BinanceDataLoader:
    def __init__(
        self, 
        backend: str = "local",
        path: str = "./Data",
        ...
    ):
        self.storage = get_storage_backend(backend, path)
        self.cache = BarCache(self.storage)
```

## S3 後端實作

### S3StorageBackend

```python
# storage/s3.py
class S3StorageBackend(StorageBackend):
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix
        self._check_boto3()  # 檢查 optional dependency
    
    def _full_path(self, path: str) -> str:
        """組合完整 S3 URI 供 DuckDB 使用"""
        if self.prefix:
            return f"s3://{self.bucket}/{self.prefix}/{path}"
        return f"s3://{self.bucket}/{path}"
    
    def read_parquet(self, path: str) -> pl.DataFrame:
        import duckdb
        con = duckdb.connect(":memory:")
        return con.execute(
            f"SELECT * FROM read_parquet('{self._full_path(path)}')"
        ).pl()
    
    def glob(self, pattern: str) -> List[str]:
        """回傳 S3 URI 格式的路徑列表"""
        # 使用 boto3 列出物件，或利用 DuckDB 的 glob 功能
        ...
```

### DuckDB S3 認證

DuckDB 會自動讀取以下環境變數：
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

### Optional Dependency

```python
def _check_boto3(self):
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "S3 backend requires boto3. Install with: pip install factorium[s3]"
        )
```

## 技術備註

1. **以 Polars 為主**：DuckDB 讀取後透過 `.pl()` 轉成 `pl.DataFrame`
2. **路徑為相對路徑**：`path` 參數是相對於 backend 根目錄的路徑
3. **glob 支援**：DuckDB 需要 glob pattern 來讀取 Hive 分區，S3 後端需將 `s3://` 前綴加回去
4. **向後相容**：現有使用預設參數的程式碼會自動使用 Local 後端

## 下一步

1. 在 `dev` 分支建立 feature branch
2. 使用 TDD 方式實作 `storage/` 模組
3. 修改 `BarCache` 和 `BinanceDataLoader`
4. 更新文檔
