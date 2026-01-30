# tests/data/test_timestamp_utils.py
from factorium.data.loader import _detect_timestamp_unit, _convert_to_target_unit


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
