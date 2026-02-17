# src/factorium/storage/local.py
"""Local filesystem storage backend."""

from pathlib import Path

import polars as pl

from .base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Storage backend for local filesystem."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative path to absolute path within base_path.

        Raises:
            ValueError: If path is absolute or attempts directory traversal
        """
        # Prevent absolute path override
        if Path(path).is_absolute():
            raise ValueError(f"Path must be relative, got: {path}")

        resolved = (self.base_path / path).resolve()
        base_resolved = self.base_path.resolve()

        # Ensure the resolved path is within the base path (cross-platform)
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise ValueError(f"Path traversal detected: {path}")

        return resolved

    def full_path(self, path: str) -> str:
        """Get absolute path string for DuckDB queries."""
        return str(self._resolve_path(path).resolve())

    def read_parquet(self, path: str) -> pl.DataFrame:
        return pl.read_parquet(self._resolve_path(path))

    def write_parquet(self, df: pl.DataFrame, path: str) -> None:
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(full_path)

    def exists(self, path: str) -> bool:
        return self._resolve_path(path).exists()

    def glob(self, pattern: str) -> list[str]:
        matches = list(self.base_path.glob(pattern))
        return [str(m.relative_to(self.base_path)) for m in matches]

    def delete(self, path: str) -> None:
        self._resolve_path(path).unlink(missing_ok=True)

    def makedirs(self, path: str) -> None:
        self._resolve_path(path).mkdir(parents=True, exist_ok=True)
