# tests/storage/test_base.py
import pytest
from factorium.storage.base import StorageBackend


def test_storage_backend_cannot_be_instantiated():
    """StorageBackend is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        StorageBackend()
