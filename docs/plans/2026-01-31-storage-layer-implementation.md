# Storage Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 建立 Storage Layer 抽象層，讓用戶可以選擇將資料儲存在本地端或 S3。

**Architecture:** 使用 Strategy Pattern 建立 `StorageBackend` 抽象基類，實作 `LocalStorageBackend` 和 `S3StorageBackend`。通過 Factory 函數 `get_storage_backend()` 建立實例。修改 `BarCache` 和 `BinanceDataLoader` 使用注入的 `StorageBackend`。

**Tech Stack:** Python 3.13, Polars, DuckDB, boto3 (optional), pytest

---

## Task 1: 建立 StorageBackend 抽象基類

**Files:**
- Create: `src/factorium/storage/__init__.py`
- Create: `src/factorium/storage/base.py`
- Test: `tests/storage/test_base.py`

**Step 1: 建立 storage 目錄結構**

```bash
mkdir -p src/factorium/storage tests/storage
touch src/factorium/storage/__init__.py tests/storage/__init__.py
```

**Step 2: 寫測試確認抽象類無法直接實例化**

```python
# tests/storage/test_base.py
import pytest
from factorium.storage.base import StorageBackend

def test_storage_backend_cannot_be_instantiated():
    """StorageBackend is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        StorageBackend()
```

**Step 3: 執行測試確認失敗**

```bash
uv run pytest tests/storage/test_base.py -v
```
Expected: FAIL (module not found)

**Step 4: 實作抽象基類**

```python
# src/factorium/storage/base.py
"""Storage backend abstraction layer."""

from abc import ABC, abstractmethod
from typing import List

import polars as pl


class StorageBackend(ABC):
    """Abstract base class for storage backends.
    
    All storage backends must implement these methods to support
    reading and writing Parquet files across different storage systems.
    """

    @abstractmethod
    def read_parquet(self, path: str) -> pl.DataFrame:
        """Read a single Parquet file.
        
        Args:
            path: Relative path to the Parquet file
            
        Returns:
            Polars DataFrame with file contents
        """
        ...

    @abstractmethod
    def write_parquet(self, df: pl.DataFrame, path: str) -> None:
        """Write a Polars DataFrame to a Parquet file.
        
        Args:
            df: DataFrame to write
            path: Relative path for the output file
        """
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a path exists.
        
        Args:
            path: Relative path to check
            
        Returns:
            True if path exists, False otherwise
        """
        ...

    @abstractmethod
    def glob(self, pattern: str) -> List[str]:
        """List files matching a glob pattern.
        
        Args:
            pattern: Glob pattern to match
            
        Returns:
            List of matching file paths
        """
        ...

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file.
        
        Args:
            path: Relative path to delete
        """
        ...

    @abstractmethod
    def makedirs(self, path: str) -> None:
        """Create directory structure.
        
        Args:
            path: Relative path for directory to create
        """
        ...
```

**Step 5: 更新 `__init__.py`**

```python
# src/factorium/storage/__init__.py
"""Storage backend abstraction layer."""

from .base import StorageBackend

__all__ = ["StorageBackend"]
```

**Step 6: 執行測試確認通過**

```bash
uv run pytest tests/storage/test_base.py -v
```
Expected: PASS

**Step 7: Commit**

```bash
git add src/factorium/storage/ tests/storage/
git commit -m "feat(storage): add StorageBackend abstract base class"
```

---

## Task 2: 實作 LocalStorageBackend

**Files:**
- Create: `src/factorium/storage/local.py`
- Test: `tests/storage/test_local.py`

**Step 1: 寫測試**

```python
# tests/storage/test_local.py
import tempfile
from pathlib import Path

import polars as pl
import pytest

from factorium.storage.local import LocalStorageBackend


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def backend(self, temp_dir):
        """Create a LocalStorageBackend instance."""
        return LocalStorageBackend(str(temp_dir))

    def test_init_creates_base_path(self, temp_dir):
        """Backend should create base_path if it doesn't exist."""
        new_path = temp_dir / "new_dir"
        backend = LocalStorageBackend(str(new_path))
        assert new_path.exists()

    def test_write_and_read_parquet(self, backend, temp_dir):
        """Should write and read Parquet files correctly."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        backend.write_parquet(df, "test/data.parquet")
        
        result = backend.read_parquet("test/data.parquet")
        assert result.equals(df)

    def test_exists_returns_true_for_existing_file(self, backend, temp_dir):
        """exists() should return True for existing files."""
        df = pl.DataFrame({"a": [1]})
        backend.write_parquet(df, "exists.parquet")
        
        assert backend.exists("exists.parquet") is True

    def test_exists_returns_false_for_missing_file(self, backend):
        """exists() should return False for non-existing files."""
        assert backend.exists("missing.parquet") is False

    def test_glob_finds_matching_files(self, backend):
        """glob() should find files matching pattern."""
        df = pl.DataFrame({"a": [1]})
        backend.write_parquet(df, "dir1/file1.parquet")
        backend.write_parquet(df, "dir1/file2.parquet")
        backend.write_parquet(df, "dir2/file3.parquet")
        
        matches = backend.glob("dir1/*.parquet")
        assert len(matches) == 2
        assert all("dir1" in m for m in matches)

    def test_delete_removes_file(self, backend):
        """delete() should remove the file."""
        df = pl.DataFrame({"a": [1]})
        backend.write_parquet(df, "to_delete.parquet")
        assert backend.exists("to_delete.parquet")
        
        backend.delete("to_delete.parquet")
        assert not backend.exists("to_delete.parquet")

    def test_makedirs_creates_directory(self, backend, temp_dir):
        """makedirs() should create directory structure."""
        backend.makedirs("path/to/dir")
        assert (temp_dir / "path" / "to" / "dir").is_dir()

    def test_full_path_returns_absolute_path(self, backend, temp_dir):
        """full_path() should return absolute path for DuckDB."""
        full = backend.full_path("some/file.parquet")
        assert str(temp_dir) in full
        assert "some/file.parquet" in full
```

**Step 2: 執行測試確認失敗**

```bash
uv run pytest tests/storage/test_local.py -v
```
Expected: FAIL (module not found)

**Step 3: 實作 LocalStorageBackend**

```python
# src/factorium/storage/local.py
"""Local filesystem storage backend."""

from pathlib import Path
from typing import List

import polars as pl

from .base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Storage backend for local filesystem.
    
    All paths are relative to the base_path specified at initialization.
    
    Args:
        base_path: Root directory for all storage operations
        
    Example:
        >>> backend = LocalStorageBackend("./Data")
        >>> backend.write_parquet(df, "cache/data.parquet")
        >>> df = backend.read_parquet("cache/data.parquet")
    """

    def __init__(self, base_path: str):
        """Initialize local storage backend.
        
        Args:
            base_path: Root directory for storage. Created if doesn't exist.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path to absolute path."""
        return self.base_path / path

    def full_path(self, path: str) -> str:
        """Get absolute path string for DuckDB queries.
        
        Args:
            path: Relative path
            
        Returns:
            Absolute path as string
        """
        return str(self._resolve_path(path))

    def read_parquet(self, path: str) -> pl.DataFrame:
        """Read a Parquet file from local filesystem."""
        return pl.read_parquet(self._resolve_path(path))

    def write_parquet(self, df: pl.DataFrame, path: str) -> None:
        """Write a Polars DataFrame to local filesystem."""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(full_path)

    def exists(self, path: str) -> bool:
        """Check if path exists on local filesystem."""
        return self._resolve_path(path).exists()

    def glob(self, pattern: str) -> List[str]:
        """List files matching glob pattern.
        
        Returns paths relative to base_path.
        """
        matches = list(self.base_path.glob(pattern))
        return [str(m.relative_to(self.base_path)) for m in matches]

    def delete(self, path: str) -> None:
        """Delete a file from local filesystem."""
        self._resolve_path(path).unlink()

    def makedirs(self, path: str) -> None:
        """Create directory structure on local filesystem."""
        self._resolve_path(path).mkdir(parents=True, exist_ok=True)
```

**Step 4: 更新 `__init__.py`**

```python
# src/factorium/storage/__init__.py
"""Storage backend abstraction layer."""

from .base import StorageBackend
from .local import LocalStorageBackend

__all__ = ["StorageBackend", "LocalStorageBackend"]
```

**Step 5: 執行測試確認通過**

```bash
uv run pytest tests/storage/test_local.py -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/factorium/storage/local.py tests/storage/test_local.py src/factorium/storage/__init__.py
git commit -m "feat(storage): add LocalStorageBackend implementation"
```

---

## Task 3: 新增 Factory 函數

**Files:**
- Modify: `src/factorium/storage/__init__.py`
- Test: `tests/storage/test_factory.py`

**Step 1: 寫測試**

```python
# tests/storage/test_factory.py
import tempfile
from pathlib import Path

import pytest

from factorium.storage import get_storage_backend, LocalStorageBackend


class TestGetStorageBackend:
    """Tests for get_storage_backend factory function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_local_backend_default(self, temp_dir):
        """Should return LocalStorageBackend for 'local' backend."""
        backend = get_storage_backend("local", str(temp_dir))
        assert isinstance(backend, LocalStorageBackend)

    def test_local_backend_is_default(self, temp_dir):
        """'local' should be the default backend."""
        backend = get_storage_backend(path=str(temp_dir))
        assert isinstance(backend, LocalStorageBackend)

    def test_unknown_backend_raises_error(self, temp_dir):
        """Should raise ValueError for unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_storage_backend("unknown", str(temp_dir))

    def test_s3_backend_requires_boto3(self):
        """Should raise ImportError if boto3 not installed for S3."""
        # This test will be updated when S3 backend is implemented
        # For now, just test the error message mentions s3
        with pytest.raises(ValueError, match="Unknown backend"):
            get_storage_backend("s3", "my-bucket/path")
```

**Step 2: 執行測試確認失敗**

```bash
uv run pytest tests/storage/test_factory.py -v
```
Expected: FAIL (get_storage_backend not found)

**Step 3: 實作 Factory 函數**

```python
# src/factorium/storage/__init__.py
"""Storage backend abstraction layer."""

from .base import StorageBackend
from .local import LocalStorageBackend


def get_storage_backend(backend: str = "local", path: str = "./Data") -> StorageBackend:
    """Factory function to create storage backend instances.
    
    Args:
        backend: Backend type - "local" or "s3"
        path: For local: base directory path
              For s3: "bucket-name/prefix" format
              
    Returns:
        StorageBackend instance
        
    Raises:
        ValueError: If backend type is unknown
        ImportError: If S3 backend is requested but boto3 is not installed
        
    Example:
        >>> backend = get_storage_backend("local", "./Data")
        >>> backend = get_storage_backend("s3", "my-bucket/data")
    """
    if backend == "local":
        return LocalStorageBackend(path)
    elif backend == "s3":
        # S3 backend will be implemented in Task 4
        from .s3 import S3StorageBackend
        
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return S3StorageBackend(bucket=bucket, prefix=prefix)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: 'local', 's3'")


__all__ = ["StorageBackend", "LocalStorageBackend", "get_storage_backend"]
```

**Step 4: 執行測試確認通過**

```bash
uv run pytest tests/storage/test_factory.py -v
```
Expected: PASS (3 pass, 1 xfail for s3)

**Step 5: Commit**

```bash
git add src/factorium/storage/__init__.py tests/storage/test_factory.py
git commit -m "feat(storage): add get_storage_backend factory function"
```

---

## Task 4: 實作 S3StorageBackend

**Files:**
- Create: `src/factorium/storage/s3.py`
- Modify: `pyproject.toml` (add optional dependency)
- Test: `tests/storage/test_s3.py`

**Step 1: 新增 optional dependency**

在 `pyproject.toml` 的 `[project.optional-dependencies]` 加入：

```toml
[project.optional-dependencies]
s3 = ["boto3>=1.26.0"]
```

**Step 2: 寫測試 (使用 mock)**

```python
# tests/storage/test_s3.py
"""Tests for S3StorageBackend.

These tests use mocks since we don't want to require actual S3 access.
"""
import pytest
from unittest.mock import MagicMock, patch
import polars as pl


class TestS3StorageBackendImport:
    """Test S3 backend import behavior."""

    def test_import_error_without_boto3(self):
        """Should raise ImportError with helpful message when boto3 missing."""
        with patch.dict("sys.modules", {"boto3": None}):
            # Force reimport
            import importlib
            import sys
            
            # Remove cached module if exists
            if "factorium.storage.s3" in sys.modules:
                del sys.modules["factorium.storage.s3"]
            
            # This should raise ImportError
            with pytest.raises(ImportError, match="pip install factorium\\[s3\\]"):
                from factorium.storage.s3 import S3StorageBackend


class TestS3StorageBackend:
    """Tests for S3StorageBackend with mocked boto3."""

    @pytest.fixture
    def mock_boto3(self):
        """Mock boto3 module."""
        with patch("factorium.storage.s3.boto3") as mock:
            yield mock

    @pytest.fixture
    def mock_duckdb(self):
        """Mock duckdb for read operations."""
        with patch("factorium.storage.s3.duckdb") as mock:
            yield mock

    @pytest.fixture
    def backend(self, mock_boto3):
        """Create S3StorageBackend with mocked boto3."""
        from factorium.storage.s3 import S3StorageBackend
        return S3StorageBackend(bucket="test-bucket", prefix="data")

    def test_full_path_with_prefix(self, backend):
        """full_path should return correct S3 URI with prefix."""
        assert backend.full_path("cache/file.parquet") == "s3://test-bucket/data/cache/file.parquet"

    def test_full_path_without_prefix(self, mock_boto3):
        """full_path should work without prefix."""
        from factorium.storage.s3 import S3StorageBackend
        backend = S3StorageBackend(bucket="test-bucket", prefix="")
        assert backend.full_path("file.parquet") == "s3://test-bucket/file.parquet"

    def test_exists_returns_true_for_existing_object(self, backend, mock_boto3):
        """exists() should return True when object exists."""
        mock_boto3.client.return_value.head_object.return_value = {}
        assert backend.exists("existing.parquet") is True

    def test_exists_returns_false_for_missing_object(self, backend, mock_boto3):
        """exists() should return False when object doesn't exist."""
        from botocore.exceptions import ClientError
        mock_boto3.client.return_value.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )
        assert backend.exists("missing.parquet") is False

    def test_delete_calls_delete_object(self, backend, mock_boto3):
        """delete() should call S3 delete_object."""
        backend.delete("to_delete.parquet")
        mock_boto3.client.return_value.delete_object.assert_called_once()

    def test_makedirs_is_noop(self, backend):
        """makedirs() should be a no-op for S3 (directories are virtual)."""
        # Should not raise
        backend.makedirs("some/path")
```

**Step 3: 執行測試確認失敗**

```bash
uv run pytest tests/storage/test_s3.py -v
```
Expected: FAIL (module not found)

**Step 4: 實作 S3StorageBackend**

```python
# src/factorium/storage/s3.py
"""S3 storage backend using DuckDB for Parquet operations."""

from typing import List
import io

import polars as pl

from .base import StorageBackend

# Check for boto3 at import time
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    raise ImportError(
        "S3 backend requires boto3. Install with: pip install factorium[s3]"
    )


class S3StorageBackend(StorageBackend):
    """Storage backend for Amazon S3.
    
    Uses boto3 for S3 operations and DuckDB for reading Parquet files.
    DuckDB natively supports S3 URIs when AWS credentials are configured
    via environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION).
    
    Args:
        bucket: S3 bucket name
        prefix: Optional prefix (folder path) within the bucket
        
    Example:
        >>> backend = S3StorageBackend(bucket="my-bucket", prefix="data")
        >>> # Reads from s3://my-bucket/data/cache/file.parquet
        >>> df = backend.read_parquet("cache/file.parquet")
    """

    def __init__(self, bucket: str, prefix: str = ""):
        """Initialize S3 storage backend.
        
        Args:
            bucket: S3 bucket name
            prefix: Optional prefix path within bucket
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._s3_client = boto3.client("s3")

    def _build_key(self, path: str) -> str:
        """Build full S3 key from relative path."""
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def full_path(self, path: str) -> str:
        """Get full S3 URI for DuckDB queries.
        
        Args:
            path: Relative path
            
        Returns:
            Full S3 URI (s3://bucket/prefix/path)
        """
        key = self._build_key(path)
        return f"s3://{self.bucket}/{key}"

    def read_parquet(self, path: str) -> pl.DataFrame:
        """Read a Parquet file from S3 using DuckDB.
        
        DuckDB handles S3 authentication via environment variables.
        """
        import duckdb
        
        uri = self.full_path(path)
        con = duckdb.connect(":memory:")
        result = con.execute(f"SELECT * FROM read_parquet('{uri}')").pl()
        con.close()
        return result

    def write_parquet(self, df: pl.DataFrame, path: str) -> None:
        """Write a Polars DataFrame to S3.
        
        Writes to a buffer first, then uploads to S3.
        """
        key = self._build_key(path)
        
        # Write to buffer
        buffer = io.BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)
        
        # Upload to S3
        self._s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer.getvalue()
        )

    def exists(self, path: str) -> bool:
        """Check if object exists in S3."""
        key = self._build_key(path)
        try:
            self._s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def glob(self, pattern: str) -> List[str]:
        """List objects matching pattern in S3.
        
        Note: S3 doesn't support true glob patterns, so this uses prefix listing
        and filters results. For complex patterns, consider using DuckDB's glob.
        
        Returns:
            List of matching keys relative to prefix
        """
        import fnmatch
        
        # Extract prefix from pattern for efficient listing
        prefix_parts = []
        for part in pattern.split("/"):
            if "*" in part or "?" in part:
                break
            prefix_parts.append(part)
        
        list_prefix = self._build_key("/".join(prefix_parts)) if prefix_parts else self.prefix
        
        # List objects
        paginator = self._s3_client.get_paginator("list_objects_v2")
        matches = []
        
        for page in paginator.paginate(Bucket=self.bucket, Prefix=list_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Remove prefix to get relative path
                if self.prefix:
                    rel_path = key[len(self.prefix) + 1:] if key.startswith(self.prefix + "/") else key
                else:
                    rel_path = key
                
                if fnmatch.fnmatch(rel_path, pattern):
                    matches.append(rel_path)
        
        return matches

    def delete(self, path: str) -> None:
        """Delete an object from S3."""
        key = self._build_key(path)
        self._s3_client.delete_object(Bucket=self.bucket, Key=key)

    def makedirs(self, path: str) -> None:
        """No-op for S3 (directories are virtual).
        
        S3 uses a flat namespace, so directories don't need to be explicitly created.
        """
        pass
```

**Step 5: 更新 `__init__.py`**

```python
# src/factorium/storage/__init__.py
"""Storage backend abstraction layer."""

from .base import StorageBackend
from .local import LocalStorageBackend


def get_storage_backend(backend: str = "local", path: str = "./Data") -> StorageBackend:
    """Factory function to create storage backend instances.
    
    Args:
        backend: Backend type - "local" or "s3"
        path: For local: base directory path
              For s3: "bucket-name/prefix" format
              
    Returns:
        StorageBackend instance
        
    Raises:
        ValueError: If backend type is unknown
        ImportError: If S3 backend is requested but boto3 is not installed
        
    Example:
        >>> backend = get_storage_backend("local", "./Data")
        >>> backend = get_storage_backend("s3", "my-bucket/data")
    """
    if backend == "local":
        return LocalStorageBackend(path)
    elif backend == "s3":
        from .s3 import S3StorageBackend
        
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return S3StorageBackend(bucket=bucket, prefix=prefix)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: 'local', 's3'")


__all__ = ["StorageBackend", "LocalStorageBackend", "get_storage_backend"]
```

**Step 6: 執行測試確認通過**

```bash
uv run pytest tests/storage/test_s3.py -v
```
Expected: PASS (with mocks)

**Step 7: Commit**

```bash
git add src/factorium/storage/s3.py tests/storage/test_s3.py pyproject.toml
git commit -m "feat(storage): add S3StorageBackend implementation"
```

---

## Task 5: 修改 BarCache 使用 StorageBackend

**Files:**
- Modify: `src/factorium/data/cache.py`
- Modify: `tests/data/test_cache.py`

**Step 1: 更新測試使用 StorageBackend**

```python
# tests/data/test_cache.py - 更新現有測試
# 需要先閱讀現有測試然後修改
```

**Step 2: 修改 BarCache 類別**

關鍵改動：
1. 將 `__init__` 參數從 `cache_dir: Path` 改為 `storage: StorageBackend, cache_prefix: str = ".cache"`
2. 將所有 `Path` 操作改為 `storage` 方法調用
3. 保持向後相容：如果傳入 `Path` 或 `str`，自動建立 `LocalStorageBackend`

```python
# src/factorium/data/cache.py
class BarCache:
    def __init__(
        self,
        storage: StorageBackend | Path | str | None = None,
        cache_prefix: str = ".cache",
    ):
        """Initialize cache.
        
        Args:
            storage: StorageBackend instance, or path for backward compatibility.
                    If None, defaults to LocalStorageBackend("./Data")
            cache_prefix: Prefix path for cache files within storage
        """
        if storage is None:
            from ..storage import LocalStorageBackend
            self.storage = LocalStorageBackend("./Data")
        elif isinstance(storage, (str, Path)):
            # Backward compatibility: accept path string
            from ..storage import LocalStorageBackend
            self.storage = LocalStorageBackend(str(storage))
            cache_prefix = ""  # Path already includes cache dir
        else:
            self.storage = storage
        
        self.cache_prefix = cache_prefix
        self.storage.makedirs(cache_prefix)
```

**Step 3: 執行所有 cache 相關測試**

```bash
uv run pytest tests/data/test_cache.py -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add src/factorium/data/cache.py tests/data/test_cache.py
git commit -m "refactor(cache): use StorageBackend for cache operations"
```

---

## Task 6: 修改 BinanceDataLoader 支援 StorageBackend

**Files:**
- Modify: `src/factorium/data/loader.py`
- Test: `tests/data/test_loader.py`

**Step 1: 更新 BinanceDataLoader 參數**

```python
# src/factorium/data/loader.py
class BinanceDataLoader:
    def __init__(
        self,
        backend: str = "local",
        path: str = "./Data",
        # 保留舊參數以維持向後相容
        base_path: str | None = None,
        max_concurrent_downloads: int = 5,
        retry_attempts: int = 3,
        retry_delay: int = 1,
    ):
        # 向後相容：如果使用 base_path，轉換為新參數
        if base_path is not None:
            import warnings
            warnings.warn(
                "base_path is deprecated, use backend='local' and path instead",
                DeprecationWarning,
                stacklevel=2,
            )
            backend = "local"
            path = base_path
        
        from ..storage import get_storage_backend
        self.storage = get_storage_backend(backend, path)
        self.base_path = Path(path)  # 保留以相容現有程式碼
        ...
```

**Step 2: 更新 `_check_all_files_exist` 使用 storage**

**Step 3: 更新 BarCache 初始化**

```python
cache = BarCache(storage=self.storage) if (use_cache and bar_type == "time") else None
```

**Step 4: 執行測試**

```bash
uv run pytest tests/data/ -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/factorium/data/loader.py tests/data/test_loader.py
git commit -m "feat(loader): add storage backend support to BinanceDataLoader"
```

---

## Task 7: 整合測試與文檔更新

**Files:**
- Create: `tests/storage/test_integration.py`
- Modify: `docs/user-guide/data-acquisition.md`

**Step 1: 寫整合測試**

```python
# tests/storage/test_integration.py
"""Integration tests for storage backends with data loading."""
import tempfile
from pathlib import Path

import polars as pl
import pytest

from factorium.storage import get_storage_backend, LocalStorageBackend
from factorium.data.cache import BarCache


class TestStorageIntegration:
    """Integration tests for storage with cache."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cache_with_local_backend(self, temp_dir):
        """BarCache should work with LocalStorageBackend."""
        backend = LocalStorageBackend(str(temp_dir))
        cache = BarCache(storage=backend)
        
        # Create and cache data
        from datetime import datetime
        df = pl.DataFrame({
            "start_time": [1704067200000],
            "symbol": ["BTCUSDT"],
            "open": [42000.0],
        })
        
        cache.put(
            df=df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60000,
            data_type="aggTrades",
            market_type="futures_um",
            date=datetime(2024, 1, 1),
        )
        
        # Retrieve cached data
        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60000,
            data_type="aggTrades",
            market_type="futures_um",
            date=datetime(2024, 1, 1),
        )
        
        assert result is not None
        assert result.equals(df)
```

**Step 2: 執行所有測試**

```bash
uv run pytest -v
```
Expected: 所有測試通過

**Step 3: 更新文檔**

在 `docs/user-guide/data-acquisition.md` 加入 Storage Backend 使用說明。

**Step 4: Commit**

```bash
git add tests/storage/test_integration.py docs/
git commit -m "docs: add storage backend documentation and integration tests"
```

---

## Task 8: 最終驗證與清理

**Step 1: 執行完整測試套件**

```bash
uv run pytest -v --tb=short
```
Expected: 所有測試通過

**Step 2: 執行 type checking (如有)**

```bash
uv run mypy src/factorium/storage/
```

**Step 3: 更新 pyproject.toml 版本 (如需要)**

**Step 4: 最終 commit**

```bash
git add -A
git commit -m "chore: finalize storage layer implementation"
```

---

## Summary

| Task | 描述 | 預估時間 |
|------|------|----------|
| 1 | StorageBackend 抽象基類 | 5 min |
| 2 | LocalStorageBackend 實作 | 10 min |
| 3 | Factory 函數 | 5 min |
| 4 | S3StorageBackend 實作 | 15 min |
| 5 | 修改 BarCache | 10 min |
| 6 | 修改 BinanceDataLoader | 15 min |
| 7 | 整合測試與文檔 | 10 min |
| 8 | 最終驗證 | 5 min |

**Total: ~75 分鐘**
