# tests/storage/test_factory.py
import tempfile
from pathlib import Path
from unittest.mock import patch

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

    def test_s3_backend_creates_instance(self):
        """S3 backend should create S3StorageBackend instance."""
        with patch("factorium.storage.s3.boto3"):
            from factorium.storage.s3 import S3StorageBackend

            backend = get_storage_backend("s3", "my-bucket/path")
            assert isinstance(backend, S3StorageBackend)
