"""Tests for AggBarMetadata."""

import pytest

from factorium.data.metadata import AggBarMetadata


class TestAggBarMetadata:
    """Test suite for AggBarMetadata dataclass."""

    def test_create_metadata(self):
        """Test basic creation with symbols, min_time, max_time, num_rows."""
        metadata = AggBarMetadata(
            symbols=["AAPL", "MSFT"],
            min_time=1609459200000,
            max_time=1609545600000,
            num_rows=100,
        )

        assert metadata.symbols == ["AAPL", "MSFT"]
        assert metadata.min_time == 1609459200000
        assert metadata.max_time == 1609545600000
        assert metadata.num_rows == 100

    def test_merge_two_metadata(self):
        """Test merging two metadata objects (same symbol)."""
        metadata1 = AggBarMetadata(
            symbols=["AAPL"],
            min_time=1609459200000,
            max_time=1609470000000,
            num_rows=50,
        )
        metadata2 = AggBarMetadata(
            symbols=["AAPL"],
            min_time=1609470000000,
            max_time=1609545600000,
            num_rows=50,
        )

        merged = AggBarMetadata.merge([metadata1, metadata2])

        assert merged.symbols == ["AAPL"]
        assert merged.min_time == 1609459200000
        assert merged.max_time == 1609545600000
        assert merged.num_rows == 100

    def test_merge_with_different_symbols(self):
        """Test merging with different symbols."""
        metadata1 = AggBarMetadata(
            symbols=["AAPL"],
            min_time=1609459200000,
            max_time=1609470000000,
            num_rows=50,
        )
        metadata2 = AggBarMetadata(
            symbols=["MSFT", "GOOG"],
            min_time=1609470000000,
            max_time=1609545600000,
            num_rows=50,
        )

        merged = AggBarMetadata.merge([metadata1, metadata2])

        assert merged.symbols == ["AAPL", "MSFT", "GOOG"]
        assert merged.min_time == 1609459200000
        assert merged.max_time == 1609545600000
        assert merged.num_rows == 100

    def test_merge_empty_list_raises(self):
        """Test that merging empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot merge empty metadata list"):
            AggBarMetadata.merge([])

    def test_merge_single_returns_copy(self):
        """Test that merging single metadata returns equivalent copy."""
        original = AggBarMetadata(
            symbols=["AAPL"],
            min_time=1609459200000,
            max_time=1609545600000,
            num_rows=100,
        )

        merged = AggBarMetadata.merge([original])

        assert merged.symbols == original.symbols
        assert merged.min_time == original.min_time
        assert merged.max_time == original.max_time
        assert merged.num_rows == original.num_rows
        # Verify it's a different object (true copy)
        assert merged is not original
        # Verify symbols list is a different object
        assert merged.symbols is not original.symbols
