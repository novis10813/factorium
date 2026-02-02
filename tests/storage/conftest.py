"""Fixtures for storage tests."""

import os
import pytest

# MinIO test settings
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_TEST_BUCKET = os.environ.get("MINIO_TEST_BUCKET", "factorium-test")


try:
    from botocore.exceptions import EndpointConnectionError
except ImportError:
    EndpointConnectionError = Exception

def is_minio_available() -> bool:
    """Check if MinIO is available for testing."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client(
            "s3",
            endpoint_url=f"http://{MINIO_ENDPOINT}",
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
        )
        s3.list_buckets()
        return True
    except (ImportError, EndpointConnectionError, Exception):
        return False


# Skip marker for tests requiring MinIO
requires_minio = pytest.mark.skipif(not is_minio_available(), reason="MinIO not available")


@pytest.fixture(scope="session")
def minio_client():
    """Create a boto3 client connected to MinIO."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


@pytest.fixture(scope="session")
def minio_test_bucket(minio_client):
    """Create and return test bucket, clean up after tests."""
    bucket = MINIO_TEST_BUCKET

    # Create bucket if not exists
    try:
        minio_client.head_bucket(Bucket=bucket)
    except:
        minio_client.create_bucket(Bucket=bucket)

    yield bucket

    # Cleanup: delete all objects and bucket
    try:
        paginator = minio_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                minio_client.delete_object(Bucket=bucket, Key=obj["Key"])
        minio_client.delete_bucket(Bucket=bucket)
    except:
        pass  # Ignore cleanup errors
