# src/factorium/storage/s3.py
"""S3 storage backend using DuckDB for Parquet operations."""

from typing import List
import io
import os

import polars as pl

from .base import StorageBackend

# Check for boto3 at import time
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    raise ImportError("S3 backend requires boto3. Install with: pip install factorium[s3]")


class S3StorageBackend(StorageBackend):
    """Storage backend for Amazon S3.

    Supports custom S3-compatible endpoints (MinIO, LocalStack) via AWS_ENDPOINT_URL.
    """

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.strip("/")

        # Support custom endpoint for MinIO/LocalStack
        self._endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        if self._endpoint_url:
            self._s3_client = boto3.client("s3", endpoint_url=self._endpoint_url)
        else:
            self._s3_client = boto3.client("s3")

    def _build_key(self, path: str) -> str:
        """Build full S3 key from relative path."""
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path

    def full_path(self, path: str) -> str:
        """Get full S3 URI for DuckDB queries."""
        key = self._build_key(path)
        return f"s3://{self.bucket}/{key}"

    def _configure_duckdb_s3(self, con) -> None:
        """Configure DuckDB connection for S3 access.

        DuckDB requires explicit S3 configuration, it doesn't read AWS_ENDPOINT_URL.
        """
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        region = os.environ.get("AWS_REGION", "us-east-1")

        con.execute(f"SET s3_access_key_id='{access_key}'")
        con.execute(f"SET s3_secret_access_key='{secret_key}'")
        con.execute(f"SET s3_region='{region}'")

        if self._endpoint_url:
            # For MinIO/LocalStack: configure custom endpoint
            # Remove http:// or https:// prefix for DuckDB
            endpoint = self._endpoint_url.replace("http://", "").replace("https://", "")
            con.execute(f"SET s3_endpoint='{endpoint}'")
            con.execute("SET s3_use_ssl=false")
            con.execute("SET s3_url_style='path'")

    def read_parquet(self, path: str) -> pl.DataFrame:
        """Read a Parquet file from S3 using DuckDB.

        Note:
            Requires AWS credentials to be configured via environment variables
            (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION) or ~/.aws/credentials.
            For MinIO/LocalStack, set AWS_ENDPOINT_URL.
        """
        import duckdb

        uri = self.full_path(path)
        con = duckdb.connect(":memory:")
        try:
            self._configure_duckdb_s3(con)
            result = con.execute(f"SELECT * FROM read_parquet('{uri}')").pl()
        finally:
            con.close()
        return result

    def write_parquet(self, df: pl.DataFrame, path: str) -> None:
        """Write a Polars DataFrame to S3."""
        key = self._build_key(path)

        buffer = io.BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)

        self._s3_client.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())

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
        """List objects matching pattern in S3."""
        import fnmatch

        prefix_parts = []
        for part in pattern.split("/"):
            if "*" in part or "?" in part:
                break
            prefix_parts.append(part)

        list_prefix = self._build_key("/".join(prefix_parts)) if prefix_parts else self.prefix

        paginator = self._s3_client.get_paginator("list_objects_v2")
        matches = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=list_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.prefix:
                    rel_path = key[len(self.prefix) + 1 :] if key.startswith(self.prefix + "/") else key
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
        """No-op for S3 (directories are virtual)."""
        pass
