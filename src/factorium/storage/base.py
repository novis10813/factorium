# src/factorium/storage/base.py
"""Storage backend abstraction layer."""

from abc import ABC, abstractmethod

import polars as pl


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    All storage backends must implement these methods to support
    reading and writing Parquet files across different storage systems.
    """

    @abstractmethod
    def full_path(self, path: str) -> str:
        """Get absolute path string for DuckDB queries."""
        ...

    @abstractmethod
    def read_parquet(self, path: str) -> pl.DataFrame:
        """Read a single Parquet file."""
        ...

    @abstractmethod
    def write_parquet(self, df: pl.DataFrame, path: str) -> None:
        """Write a Polars DataFrame to a Parquet file."""
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        ...

    @abstractmethod
    def glob(self, pattern: str) -> list[str]:
        """List files matching a glob pattern."""
        ...

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file."""
        ...

    @abstractmethod
    def makedirs(self, path: str) -> None:
        """Create directory structure."""
        ...
