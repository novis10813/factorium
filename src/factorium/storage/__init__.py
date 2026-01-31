# src/factorium/storage/__init__.py
"""Storage backend abstraction layer."""

from .base import StorageBackend
from .local import LocalStorageBackend

__all__ = ["StorageBackend", "LocalStorageBackend"]
