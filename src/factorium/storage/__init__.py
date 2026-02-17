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
        try:
            from .s3 import S3StorageBackend
        except ImportError:
            raise ImportError("S3 backend requires boto3. Install with: pip install factorium[s3]")

        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return S3StorageBackend(bucket=bucket, prefix=prefix)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: 'local', 's3'")


__all__ = ["StorageBackend", "LocalStorageBackend", "get_storage_backend"]
