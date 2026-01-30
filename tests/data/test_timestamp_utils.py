# tests/data/test_timestamp_utils.py
import polars as pl
import pytest

from factorium.data.loader import (
    _convert_to_target_unit,
    _detect_timestamp_unit,
    _normalize_timestamps_to_ms,
)


def test_detect_timestamp_unit_seconds():
    assert _detect_timestamp_unit(1704067200) == "s"


def test_detect_timestamp_unit_milliseconds():
    assert _detect_timestamp_unit(1704067200000) == "ms"


def test_detect_timestamp_unit_microseconds():
    assert _detect_timestamp_unit(1704067200000000) == "us"


def test_detect_timestamp_unit_nanoseconds():
    assert _detect_timestamp_unit(1704067200000000000) == "ns"


def test_convert_to_target_unit():
    ts_ms = 1704067200000
    assert _convert_to_target_unit(ts_ms, "s") == 1704067200
    assert _convert_to_target_unit(ts_ms, "ms") == 1704067200000
    assert _convert_to_target_unit(ts_ms, "us") == 1704067200000000
    assert _convert_to_target_unit(ts_ms, "ns") == 1704067200000000000


def test_convert_to_target_unit_invalid_unit():
    """Verify that invalid unit raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported target unit"):
        _convert_to_target_unit(1704067200000, "invalid")


class TestNormalizeTimestampsToMs:
    """Tests for _normalize_timestamps_to_ms function."""

    def test_normalize_from_microseconds(self):
        """Verify microsecond timestamps are converted to milliseconds."""
        df = pl.DataFrame(
            {
                "start_time": [1704067200000000, 1704067260000000],  # microseconds
                "end_time": [1704067259999000, 1704067319999000],
            }
        )
        result = _normalize_timestamps_to_ms(df, "us")
        assert result["start_time"][0] == 1704067200000  # milliseconds
        assert result["end_time"][0] == 1704067259999

    def test_normalize_from_nanoseconds(self):
        """Verify nanosecond timestamps are converted to milliseconds."""
        df = pl.DataFrame(
            {
                "start_time": [1704067200000000000, 1704067260000000000],  # nanoseconds
                "end_time": [1704067259999000000, 1704067319999000000],
            }
        )
        result = _normalize_timestamps_to_ms(df, "ns")
        assert result["start_time"][0] == 1704067200000  # milliseconds

    def test_normalize_from_seconds(self):
        """Verify second timestamps are converted to milliseconds."""
        df = pl.DataFrame(
            {
                "start_time": [1704067200, 1704067260],  # seconds
                "end_time": [1704067259, 1704067319],
            }
        )
        result = _normalize_timestamps_to_ms(df, "s")
        assert result["start_time"][0] == 1704067200000  # milliseconds

    def test_normalize_milliseconds_unchanged(self):
        """Verify millisecond timestamps are returned unchanged."""
        df = pl.DataFrame(
            {
                "start_time": [1704067200000, 1704067260000],  # milliseconds
                "end_time": [1704067259999, 1704067319999],
            }
        )
        result = _normalize_timestamps_to_ms(df, "ms")
        assert result["start_time"][0] == 1704067200000  # unchanged

    def test_normalize_invalid_unit(self):
        """Verify that invalid unit raises ValueError."""
        df = pl.DataFrame(
            {
                "start_time": [1704067200000],
                "end_time": [1704067259999],
            }
        )
        with pytest.raises(ValueError, match="Unsupported timestamp unit"):
            _normalize_timestamps_to_ms(df, "invalid")
