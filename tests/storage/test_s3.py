# tests/storage/test_s3.py
"""Tests for S3StorageBackend using mocks."""

import io
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import polars as pl


class TestS3StorageBackend:
    """Tests for S3StorageBackend with mocked boto3."""

    @pytest.fixture
    def mock_boto3(self):
        """Mock boto3 module."""
        with patch("factorium.storage.s3.boto3") as mock:
            # Setup default mock behavior
            mock.client.return_value = MagicMock()
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
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        assert backend.exists("missing.parquet") is False

    def test_delete_calls_delete_object(self, backend, mock_boto3):
        """delete() should call S3 delete_object."""
        backend.delete("to_delete.parquet")
        mock_boto3.client.return_value.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="data/to_delete.parquet"
        )

    def test_makedirs_is_noop(self, backend):
        """makedirs() should be a no-op for S3."""
        # Should not raise
        backend.makedirs("some/path")

    def test_write_parquet_uploads_to_s3(self, backend, mock_boto3):
        """write_parquet() should upload parquet data to S3."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        backend.write_parquet(df, "test.parquet")

        # Verify put_object was called
        mock_boto3.client.return_value.put_object.assert_called_once()
        call_kwargs = mock_boto3.client.return_value.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "data/test.parquet"

    def test_read_parquet_uses_duckdb(self, backend, mock_boto3):
        """read_parquet() should use DuckDB with correct S3 URI."""
        import sys

        mock_duckdb = MagicMock()
        mock_con = MagicMock()
        mock_duckdb.connect.return_value = mock_con
        mock_con.execute.return_value.pl.return_value = pl.DataFrame({"a": [1]})

        with patch.dict("sys.modules", {"duckdb": mock_duckdb}):
            result = backend.read_parquet("test.parquet")

        mock_duckdb.connect.assert_called_once_with(":memory:")
        mock_con.execute.assert_called_once()
        assert "s3://test-bucket/data/test.parquet" in mock_con.execute.call_args[0][0]
        mock_con.close.assert_called_once()

    def test_glob_returns_matching_files(self, backend, mock_boto3):
        """glob() should return files matching pattern."""
        # Mock paginator response
        mock_paginator = MagicMock()
        mock_boto3.client.return_value.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "data/cache/file1.parquet"},
                    {"Key": "data/cache/file2.parquet"},
                    {"Key": "data/other/file3.parquet"},
                ]
            }
        ]

        matches = backend.glob("cache/*.parquet")

        assert len(matches) == 2
        assert "cache/file1.parquet" in matches
        assert "cache/file2.parquet" in matches
