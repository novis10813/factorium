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
