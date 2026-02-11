# Bug Fix: Klines Timestamp Unit Mismatch (#18)

**Issue**: https://github.com/novis10813/factorium/issues/18
**Date**: 2026-01-30
**Version**: 0.3.1

## Problem Summary

`BinanceDataLoader.load_aggbar` fails when loading klines data due to timestamp unit mismatch:

- **Symptom**: Query returns 0 rows when loading klines data
- **Root Cause**: `_load_klines_direct` uses milliseconds for `start_ts`/`end_ts`, but `open_time`/`close_time` in parquet may be stored in different units (microseconds, nanoseconds, etc.)
- **Evidence**: 
  - User reports: `open_time` value is `1769644800000000` (16 digits, microseconds)
  - Code generates: `start_ts` value is `1769644800000` (13 digits, milliseconds)
  - WHERE condition `open_time >= 1769644800000` matches nothing because 1.7e15 > 1.7e12

## Analysis

### Data Flow
1. Different exchanges may use different timestamp units in their source data
2. `csv_to_parquet()` writes to parquet using PyArrow
3. DuckDB reads parquet - timestamp format varies by source
4. Query uses millisecond comparisons which may not match stored format

### Affected Code
- `src/factorium/data/loader.py:648-680` - `_load_klines_direct()`
  - Line 648-649: `start_ts = int(start_dt.timestamp() * 1000)` (milliseconds)
  - Line 680: `WHERE open_time >= {start_ts} AND close_time <= {end_ts}`

## Solution: Dynamic Timestamp Detection (Recommended)

Dynamically detect the timestamp unit from the data and adapt the query accordingly. This approach:
- Works with any exchange's timestamp format
- No need to hardcode logic per exchange
- Future-proof for new data sources

### Timestamp Unit Detection Logic

```python
def _detect_timestamp_unit(sample_ts: int) -> str:
    """Detect timestamp unit based on digit count.
    
    Returns:
        'ns' for nanoseconds (19 digits)
        'us' for microseconds (16 digits)  
        'ms' for milliseconds (13 digits)
        's' for seconds (10 digits)
    """
    ts_digits = len(str(abs(sample_ts)))
    if ts_digits >= 19:
        return 'ns'
    elif ts_digits >= 16:
        return 'us'
    elif ts_digits >= 13:
        return 'ms'
    else:
        return 's'

def _convert_to_target_unit(ts_ms: int, target_unit: str) -> int:
    """Convert millisecond timestamp to target unit."""
    if target_unit == 'ns':
        return ts_ms * 1_000_000
    elif target_unit == 'us':
        return ts_ms * 1_000
    elif target_unit == 'ms':
        return ts_ms
    else:  # seconds
        return ts_ms // 1000
```

### Detection at Query Time

```python
# In _load_klines_direct():
# 1. Sample the timestamp format from data
sample_query = f"""
    SELECT open_time 
    FROM read_parquet('{parquet_glob}', hive_partitioning=true) 
    LIMIT 1
"""
sample_ts = duckdb.execute(sample_query).fetchone()[0]
ts_unit = _detect_timestamp_unit(sample_ts)

# 2. Convert query timestamps to match data format
start_ts_converted = _convert_to_target_unit(start_ts, ts_unit)
end_ts_converted = _convert_to_target_unit(end_ts, ts_unit)

# 3. Use converted timestamps in WHERE clause
WHERE open_time >= {start_ts_converted} AND close_time <= {end_ts_converted}
```

## Implementation Plan

### Task 1: Add Timestamp Detection Utility Functions

**Files**: `src/factorium/data/loader.py`

**Step 1**: Write failing tests for `_detect_timestamp_unit` and `_convert_to_target_unit`
```python
# tests/data/test_timestamp_utils.py
def test_detect_timestamp_unit_seconds():
    assert _detect_timestamp_unit(1704067200) == 's'

def test_detect_timestamp_unit_milliseconds():
    assert _detect_timestamp_unit(1704067200000) == 'ms'

def test_detect_timestamp_unit_microseconds():
    assert _detect_timestamp_unit(1704067200000000) == 'us'

def test_detect_timestamp_unit_nanoseconds():
    assert _detect_timestamp_unit(1704067200000000000) == 'ns'

def test_convert_to_target_unit():
    ts_ms = 1704067200000
    assert _convert_to_target_unit(ts_ms, 's') == 1704067200
    assert _convert_to_target_unit(ts_ms, 'ms') == 1704067200000
    assert _convert_to_target_unit(ts_ms, 'us') == 1704067200000000
    assert _convert_to_target_unit(ts_ms, 'ns') == 1704067200000000000
```
**Step 2**: Verify tests fail (RED)
**Step 3**: Implement utility functions in `loader.py`
**Step 4**: Verify tests pass (GREEN)
**Step 5**: Commit with message "feat: add timestamp unit detection utilities"

### Task 2: Add Failing Test for Dynamic Detection

**Files**: `tests/data/test_loader_klines.py`

**Step 1**: Create fixture with microsecond timestamps (matching real data)
```python
@pytest.fixture
def sample_klines_data_microseconds():
    """Create klines data with microsecond timestamps."""
    # Use 16-digit timestamps
    base_ts = int(datetime(2024, 1, 1).timestamp() * 1_000_000)  # microseconds
    # ... rest of fixture
```

**Step 2**: Write test that loads data with microsecond timestamps
```python
def test_load_aggbar_klines_auto_detects_timestamp_unit(sample_klines_data_microseconds):
    """Verify klines loading auto-detects and handles microsecond timestamps."""
    loader = BinanceDataLoader(root_path=tmpdir)
    result = loader.load_aggbar(...)
    assert len(result) > 0  # Should not be empty
```
**Step 3**: Verify test fails (RED)
**Step 4**: Commit with message "test: add klines microsecond timestamp detection test"

### Task 3: Implement Dynamic Detection in `_load_klines_direct`

**Files**: `src/factorium/data/loader.py`

**Step 1**: Modify `_load_klines_direct()` to:
  1. Sample `open_time` from parquet before main query
  2. Detect timestamp unit using `_detect_timestamp_unit()`
  3. Convert `start_ts`/`end_ts` using `_convert_to_target_unit()`
  4. Use converted values in WHERE clause

**Step 2**: Handle edge case: empty parquet files (no sample available)
**Step 3**: Verify all tests pass (GREEN)
**Step 4**: Commit with message "fix: auto-detect klines timestamp unit in query (#18)"

### Task 4: Update Result Timestamp Normalization

**Files**: `src/factorium/data/loader.py`

**Step 1**: Ensure returned `start_time` column is normalized to a consistent format
**Step 2**: Option A: Convert all to milliseconds (matches Binance convention)
**Step 3**: Option B: Convert all to datetime (more user-friendly)
**Step 4**: Add test to verify output format consistency
**Step 5**: Commit with message "fix: normalize klines output timestamp format"

### Task 5: Add Tests for Multiple Timestamp Formats

**Files**: `tests/data/test_loader_klines.py`

**Step 1**: Add parametrized test covering all timestamp units:
```python
@pytest.mark.parametrize("ts_unit,multiplier", [
    ('s', 1),
    ('ms', 1000),
    ('us', 1_000_000),
    ('ns', 1_000_000_000),
])
def test_load_aggbar_klines_handles_all_timestamp_units(ts_unit, multiplier, ...):
    """Verify klines loading handles all common timestamp formats."""
    pass
```
**Step 2**: Verify all tests pass
**Step 3**: Commit with message "test: add parametrized tests for all timestamp units"

### Task 6: Run Full Test Suite & Cleanup

**Step 1**: Run `pytest tests/` to ensure no regressions
**Step 2**: Run `mypy src/` for type checking  
**Step 3**: Run `ruff check src/` for linting
**Step 4**: Update docstrings to document the auto-detection behavior
**Step 5**: Commit with message "docs: document timestamp auto-detection behavior"

## Verification Checklist

- [ ] `_detect_timestamp_unit` correctly identifies all formats (s, ms, us, ns)
- [ ] `_convert_to_target_unit` correctly converts between formats
- [ ] `_load_klines_direct` auto-detects and handles different timestamp units
- [ ] Test with millisecond timestamps passes
- [ ] Test with microsecond timestamps passes
- [ ] Existing tests still pass
- [ ] `load_aggbar` returns expected data with consistent timestamp format
- [ ] No type errors
- [ ] Issue #18 can be closed

## Benefits of This Approach

1. **Future-proof**: Works with any exchange that uses integer timestamps
2. **No hardcoding**: No need to maintain exchange-specific timestamp logic
3. **Backward compatible**: Works with existing data regardless of format
4. **Minimal overhead**: One extra query per load operation (LIMIT 1 is fast)
5. **Self-documenting**: Clear utility functions explain the detection logic

## Notes

- Detection is based on digit count, which is reliable for Unix timestamps from 2001-2286
- Edge case: timestamps before 2001-09-09 (10 digits) may be ambiguous between seconds and short milliseconds
- For practical use cases (recent financial data), this is not a concern
