"""Integration tests for S3StorageBackend with real MinIO.

These tests require MinIO to be running. They will be skipped if MinIO is not available.

To run locally:
    docker-compose -f docker-compose.minio.yml up -d
    uv run pytest tests/storage/test_s3_integration.py -v
    docker-compose -f docker-compose.minio.yml down
"""

import os
import uuid

import polars as pl
import pytest

from .conftest import (
    requires_minio,
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
)


@requires_minio
class TestS3StorageBackendIntegration:
    """Integration tests for S3StorageBackend with real MinIO."""

    @pytest.fixture
    def s3_backend(self, minio_test_bucket, monkeypatch):
        """Create S3StorageBackend connected to MinIO."""
        # Use monkeypatch to set environment variables (auto-restored after test)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", MINIO_ACCESS_KEY)
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", MINIO_SECRET_KEY)
        monkeypatch.setenv("AWS_ENDPOINT_URL", f"http://{MINIO_ENDPOINT}")
        monkeypatch.setenv("AWS_REGION", "us-east-1")

        from factorium.storage.s3 import S3StorageBackend

        prefix = f"test-{uuid.uuid4().hex[:8]}"
        backend = S3StorageBackend(bucket=minio_test_bucket, prefix=prefix)

        yield backend

        # Cleanup: delete test prefix
        # (handled by minio_test_bucket fixture at session end)

    def test_write_and_read_parquet(self, s3_backend):
        """Should write and read Parquet files through MinIO."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
                "c": [1.1, 2.2, 3.3],
            }
        )

        s3_backend.write_parquet(df, "test/data.parquet")
        result = s3_backend.read_parquet("test/data.parquet")

        assert result.equals(df)

    def test_exists_returns_true_for_existing_object(self, s3_backend):
        """exists() should return True for existing objects in MinIO."""
        df = pl.DataFrame({"a": [1]})
        s3_backend.write_parquet(df, "exists_test.parquet")

        assert s3_backend.exists("exists_test.parquet") is True

    def test_exists_returns_false_for_missing_object(self, s3_backend):
        """exists() should return False for non-existing objects."""
        assert s3_backend.exists("nonexistent.parquet") is False

    def test_glob_finds_matching_files(self, s3_backend):
        """glob() should find files matching pattern in MinIO."""
        df = pl.DataFrame({"a": [1]})
        s3_backend.write_parquet(df, "glob_test/file1.parquet")
        s3_backend.write_parquet(df, "glob_test/file2.parquet")
        s3_backend.write_parquet(df, "other/file3.parquet")

        matches = s3_backend.glob("glob_test/*.parquet")

        assert len(matches) == 2
        assert all("glob_test" in m for m in matches)

    def test_delete_removes_object(self, s3_backend):
        """delete() should remove object from MinIO."""
        df = pl.DataFrame({"a": [1]})
        s3_backend.write_parquet(df, "delete_test.parquet")
        assert s3_backend.exists("delete_test.parquet")

        s3_backend.delete("delete_test.parquet")

        assert not s3_backend.exists("delete_test.parquet")

    def test_full_path_returns_s3_uri(self, s3_backend, minio_test_bucket):
        """full_path() should return proper S3 URI."""
        path = s3_backend.full_path("some/file.parquet")

        assert path.startswith("s3://")
        assert minio_test_bucket in path
        assert "some/file.parquet" in path

    def test_makedirs_is_noop(self, s3_backend):
        """makedirs() should not raise for S3 (virtual directories)."""
        # Should not raise
        s3_backend.makedirs("some/deep/path")

    def test_large_dataframe_roundtrip(self, s3_backend):
        """Should handle larger DataFrames correctly."""
        import numpy as np

        # Create a larger DataFrame (~1MB)
        n_rows = 50000
        df = pl.DataFrame(
            {
                "id": list(range(n_rows)),
                "value": np.random.randn(n_rows),
                "category": [f"cat_{i % 100}" for i in range(n_rows)],
            }
        )

        s3_backend.write_parquet(df, "large_test.parquet")
        result = s3_backend.read_parquet("large_test.parquet")

        assert len(result) == n_rows
        assert result.schema == df.schema
