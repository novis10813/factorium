# Polars Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate data pipeline from Pandas to Polars, eliminating intermediate conversions between DuckDB and Factor computation.

**Architecture:** 
- DuckDB outputs directly to Polars DataFrame via `.pl()`
- AggBar stores `pl.DataFrame` (eager) with pre-computed metadata
- Factor converts to `pl.LazyFrame` for computation chain
- `to_pandas()` only used for final output (plotting)

**Tech Stack:** Polars, DuckDB, Python dataclasses

---

## Task 0: Add Polars to imports in aggregator.py

**Files:**
- Modify: `src/factorium/data/aggregator.py:1-10`

**Step 1: Add polars import**

```python
# At top of file, add:
import polars as pl
```

**Step 2: Verify import works**

Run: `python -c "from factorium.data.aggregator import BarAggregator; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/factorium/data/aggregator.py
git commit -m "chore(aggregator): add polars import"
```

---

## Task 1: Create AggBarMetadata dataclass

**Files:**
- Create: `src/factorium/data/metadata.py`
- Test: `tests/data/test_metadata.py`

**Step 1: Write the failing test**

```python
# tests/data/test_metadata.py
"""Tests for AggBarMetadata."""

import pytest
from factorium.data.metadata import AggBarMetadata


class TestAggBarMetadata:
    """Tests for AggBarMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating metadata with basic values."""
        meta = AggBarMetadata(
            symbols=["BTCUSDT", "ETHUSDT"],
            min_time=1704067200000,
            max_time=1704153600000,
            num_rows=1000,
        )
        
        assert meta.symbols == ["BTCUSDT", "ETHUSDT"]
        assert meta.min_time == 1704067200000
        assert meta.max_time == 1704153600000
        assert meta.num_rows == 1000

    def test_merge_two_metadata(self):
        """Test merging two metadata objects."""
        meta1 = AggBarMetadata(
            symbols=["BTCUSDT"],
            min_time=1704067200000,
            max_time=1704153600000,
            num_rows=500,
        )
        meta2 = AggBarMetadata(
            symbols=["BTCUSDT"],
            min_time=1704153600000,
            max_time=1704240000000,
            num_rows=500,
        )
        
        merged = AggBarMetadata.merge([meta1, meta2])
        
        assert merged.symbols == ["BTCUSDT"]
        assert merged.min_time == 1704067200000
        assert merged.max_time == 1704240000000
        assert merged.num_rows == 1000

    def test_merge_with_different_symbols(self):
        """Test merging metadata with different symbols."""
        meta1 = AggBarMetadata(
            symbols=["BTCUSDT"],
            min_time=1704067200000,
            max_time=1704153600000,
            num_rows=500,
        )
        meta2 = AggBarMetadata(
            symbols=["ETHUSDT"],
            min_time=1704067200000,
            max_time=1704153600000,
            num_rows=300,
        )
        
        merged = AggBarMetadata.merge([meta1, meta2])
        
        assert set(merged.symbols) == {"BTCUSDT", "ETHUSDT"}
        assert merged.num_rows == 800

    def test_merge_empty_list_raises(self):
        """Test that merging empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot merge empty"):
            AggBarMetadata.merge([])

    def test_merge_single_returns_copy(self):
        """Test that merging single metadata returns equivalent copy."""
        meta = AggBarMetadata(
            symbols=["BTCUSDT"],
            min_time=1704067200000,
            max_time=1704153600000,
            num_rows=500,
        )
        
        merged = AggBarMetadata.merge([meta])
        
        assert merged.symbols == meta.symbols
        assert merged.min_time == meta.min_time
        assert merged.max_time == meta.max_time
        assert merged.num_rows == meta.num_rows
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_metadata.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'factorium.data.metadata'"

**Step 3: Write minimal implementation**

```python
# src/factorium/data/metadata.py
"""Metadata container for AggBar."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AggBarMetadata:
    """Immutable metadata for AggBar.
    
    Stores summary statistics computed from actual data,
    not from loader parameters.
    
    Attributes:
        symbols: List of unique symbols in the data
        min_time: Minimum start_time in milliseconds
        max_time: Maximum end_time in milliseconds
        num_rows: Total number of rows
    """
    symbols: list[str]
    min_time: int
    max_time: int
    num_rows: int

    @classmethod
    def merge(cls, metadata_list: list["AggBarMetadata"]) -> "AggBarMetadata":
        """Merge multiple metadata objects into one.
        
        Args:
            metadata_list: List of AggBarMetadata to merge
            
        Returns:
            New AggBarMetadata with combined statistics
            
        Raises:
            ValueError: If metadata_list is empty
        """
        if not metadata_list:
            raise ValueError("Cannot merge empty metadata list")
        
        if len(metadata_list) == 1:
            m = metadata_list[0]
            return cls(
                symbols=list(m.symbols),
                min_time=m.min_time,
                max_time=m.max_time,
                num_rows=m.num_rows,
            )
        
        # Collect all unique symbols (preserve order, deduplicate)
        seen = set()
        all_symbols = []
        for m in metadata_list:
            for s in m.symbols:
                if s not in seen:
                    seen.add(s)
                    all_symbols.append(s)
        
        return cls(
            symbols=all_symbols,
            min_time=min(m.min_time for m in metadata_list),
            max_time=max(m.max_time for m in metadata_list),
            num_rows=sum(m.num_rows for m in metadata_list),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_metadata.py -v`
Expected: 5 passed

**Step 5: Export from data module**

Add to `src/factorium/data/__init__.py`:
```python
from .metadata import AggBarMetadata
```

**Step 6: Commit**

```bash
git add src/factorium/data/metadata.py tests/data/test_metadata.py src/factorium/data/__init__.py
git commit -m "feat(data): add AggBarMetadata dataclass for pre-computed metadata"
```

---

## Task 2: Modify BarAggregator to return Polars + metadata

**Files:**
- Modify: `src/factorium/data/aggregator.py`
- Test: `tests/data/test_aggregator_polars.py`

**Step 1: Write the failing test**

```python
# tests/data/test_aggregator_polars.py
"""Tests for BarAggregator Polars output."""

import pytest
import polars as pl
import tempfile
from pathlib import Path
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

from factorium.data.aggregator import BarAggregator
from factorium.data.metadata import AggBarMetadata
from factorium.data.adapters.binance import BinanceAdapter


@pytest.fixture
def temp_parquet_dir():
    """Create temporary directory with test parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        
        # Create Hive-partitioned structure
        hive_path = (
            base / "market=futures_um" / "data_type=aggTrades" 
            / "symbol=BTCUSDT" / "year=2024" / "month=01" / "day=01"
        )
        hive_path.mkdir(parents=True)
        
        # Create test data
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n = 100
        base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
        
        df = pd.DataFrame({
            "agg_trade_id": np.arange(n),
            "price": 100 + np.cumsum(np.random.randn(n) * 0.1),
            "quantity": np.abs(np.random.randn(n)) + 1,
            "transact_time": base_ts + np.arange(n) * 1000,
            "is_buyer_maker": np.random.choice([True, False], n),
            "symbol": "BTCUSDT",
        })
        
        table = pa.Table.from_pandas(df)
        pq.write_table(table, hive_path / "data.parquet")
        
        yield str(base / "market=futures_um" / "data_type=aggTrades" / "**/*.parquet")


class TestAggregatorPolarsOutput:
    """Tests for Polars DataFrame output from aggregator."""

    def test_aggregate_time_bars_returns_polars_and_metadata(self, temp_parquet_dir):
        """Test that aggregate_time_bars returns (pl.DataFrame, AggBarMetadata)."""
        aggregator = BarAggregator()
        adapter = BinanceAdapter()
        column_mapping = adapter.get_column_mapping("aggTrades")
        
        result = aggregator.aggregate_time_bars(
            parquet_pattern=temp_parquet_dir,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=1704067200000,
            end_ts=1704067200000 + 100_000,
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )
        
        # Should return tuple of (DataFrame, metadata)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        df, meta = result
        assert isinstance(df, pl.DataFrame)
        assert isinstance(meta, AggBarMetadata)

    def test_metadata_reflects_actual_data(self, temp_parquet_dir):
        """Test that metadata reflects actual data, not parameters."""
        aggregator = BarAggregator()
        adapter = BinanceAdapter()
        column_mapping = adapter.get_column_mapping("aggTrades")
        
        df, meta = aggregator.aggregate_time_bars(
            parquet_pattern=temp_parquet_dir,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=1704067200000,
            end_ts=1704067200000 + 100_000,
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )
        
        # Metadata should match actual data
        assert meta.symbols == ["BTCUSDT"]
        assert meta.num_rows == len(df)
        assert meta.min_time == df["start_time"].min()
        assert meta.max_time == df["end_time"].max()

    def test_polars_dataframe_has_correct_schema(self, temp_parquet_dir):
        """Test that Polars DataFrame has expected columns."""
        aggregator = BarAggregator()
        adapter = BinanceAdapter()
        column_mapping = adapter.get_column_mapping("aggTrades")
        
        df, _ = aggregator.aggregate_time_bars(
            parquet_pattern=temp_parquet_dir,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=1704067200000,
            end_ts=1704067200000 + 100_000,
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )
        
        expected_cols = [
            "symbol", "start_time", "end_time",
            "open", "high", "low", "close", "volume", "vwap",
            "num_buyer", "num_seller", "num_buyer_volume", "num_seller_volume",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_tick_bars_returns_polars_and_metadata(self, temp_parquet_dir):
        """Test that aggregate_tick_bars returns (pl.DataFrame, AggBarMetadata)."""
        aggregator = BarAggregator()
        adapter = BinanceAdapter()
        column_mapping = adapter.get_column_mapping("aggTrades")
        
        result = aggregator.aggregate_tick_bars(
            parquet_pattern=temp_parquet_dir,
            symbol="BTCUSDT",
            interval_ticks=10,
            column_mapping=column_mapping,
            include_buyer_seller=True,
            start_ts=1704067200000,
            end_ts=1704067200000 + 100_000,
        )
        
        assert isinstance(result, tuple)
        df, meta = result
        assert isinstance(df, pl.DataFrame)
        assert isinstance(meta, AggBarMetadata)

    def test_volume_bars_returns_polars_and_metadata(self, temp_parquet_dir):
        """Test that aggregate_volume_bars returns (pl.DataFrame, AggBarMetadata)."""
        aggregator = BarAggregator()
        adapter = BinanceAdapter()
        column_mapping = adapter.get_column_mapping("aggTrades")
        
        result = aggregator.aggregate_volume_bars(
            parquet_pattern=temp_parquet_dir,
            symbol="BTCUSDT",
            interval_volume=10.0,
            column_mapping=column_mapping,
            include_buyer_seller=True,
            start_ts=1704067200000,
            end_ts=1704067200000 + 100_000,
        )
        
        assert isinstance(result, tuple)
        df, meta = result
        assert isinstance(df, pl.DataFrame)
        assert isinstance(meta, AggBarMetadata)

    def test_dollar_bars_returns_polars_and_metadata(self, temp_parquet_dir):
        """Test that aggregate_dollar_bars returns (pl.DataFrame, AggBarMetadata)."""
        aggregator = BarAggregator()
        adapter = BinanceAdapter()
        column_mapping = adapter.get_column_mapping("aggTrades")
        
        result = aggregator.aggregate_dollar_bars(
            parquet_pattern=temp_parquet_dir,
            symbol="BTCUSDT",
            interval_dollar=1000.0,
            column_mapping=column_mapping,
            include_buyer_seller=True,
            start_ts=1704067200000,
            end_ts=1704067200000 + 100_000,
        )
        
        assert isinstance(result, tuple)
        df, meta = result
        assert isinstance(df, pl.DataFrame)
        assert isinstance(meta, AggBarMetadata)

    def test_empty_result_returns_empty_dataframe_and_metadata(self, temp_parquet_dir):
        """Test that empty result returns empty DataFrame with zeroed metadata."""
        aggregator = BarAggregator()
        adapter = BinanceAdapter()
        column_mapping = adapter.get_column_mapping("aggTrades")
        
        # Query a time range with no data
        df, meta = aggregator.aggregate_time_bars(
            parquet_pattern=temp_parquet_dir,
            symbols=["NONEXISTENT"],
            interval_ms=60_000,
            start_ts=1704067200000,
            end_ts=1704067200000 + 100_000,
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )
        
        assert len(df) == 0
        assert meta.num_rows == 0
        assert meta.symbols == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_aggregator_polars.py::TestAggregatorPolarsOutput::test_aggregate_time_bars_returns_polars_and_metadata -v`
Expected: FAIL (returns pd.DataFrame, not tuple)

**Step 3: Modify aggregate_time_bars to return Polars + metadata**

In `src/factorium/data/aggregator.py`, modify `aggregate_time_bars`:

```python
def aggregate_time_bars(
    self,
    parquet_pattern: str,
    symbols: list[str],
    interval_ms: int,
    start_ts: int,
    end_ts: int,
    column_mapping: ColumnMapping,
    include_buyer_seller: bool = True,
) -> tuple[pl.DataFrame, "AggBarMetadata"]:
    """Aggregate tick data into time-based OHLCV bars.
    
    Returns:
        Tuple of (pl.DataFrame, AggBarMetadata)
    """
    from .metadata import AggBarMetadata
    
    # ... existing query building code ...
    
    try:
        conn = duckdb.connect()
        df = conn.execute(query).pl()  # Changed from .df() to .pl()
        
        if df.is_empty():
            return df, AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)
        
        # Compute metadata from actual data
        meta_row = conn.execute("""
            SELECT 
                LIST(DISTINCT symbol ORDER BY symbol) AS symbols,
                MIN(start_time) AS min_time,
                MAX(end_time) AS max_time,
                COUNT(*) AS num_rows
            FROM df
        """).fetchone()
        
        metadata = AggBarMetadata(
            symbols=meta_row[0],
            min_time=meta_row[1],
            max_time=meta_row[2],
            num_rows=meta_row[3],
        )
        
        return df, metadata
    except duckdb.IOException as e:
        logger.warning(f"DuckDB aggregation failed for pattern '{parquet_pattern}': {e}")
        return pl.DataFrame(), AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)
```

**Step 4: Apply same pattern to tick/volume/dollar bars**

Same modification for `aggregate_tick_bars`, `aggregate_volume_bars`, `aggregate_dollar_bars`.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/data/test_aggregator_polars.py -v`
Expected: All passed

**Step 6: Run existing aggregator tests to check for regressions**

Run: `pytest tests/data/test_aggregator_bars.py -v`
Expected: Some failures (tests expect pd.DataFrame)

**Step 7: Commit**

```bash
git add src/factorium/data/aggregator.py tests/data/test_aggregator_polars.py
git commit -m "feat(aggregator): return Polars DataFrame + metadata from all bar methods"
```

---

## Task 3: Migrate AggBar to use Polars DataFrame

**Files:**
- Modify: `src/factorium/aggbar.py`
- Test: `tests/test_aggbar_polars.py`

**Step 1: Write the failing test**

```python
# tests/test_aggbar_polars.py
"""Tests for AggBar with Polars backend."""

import pytest
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime

from factorium import AggBar
from factorium.data.metadata import AggBarMetadata


@pytest.fixture
def sample_polars_df():
    """Create sample Polars DataFrame for testing."""
    n = 100
    base_ts = 1704067200000
    
    return pl.DataFrame({
        "start_time": [base_ts + i * 60000 for i in range(n)],
        "end_time": [base_ts + (i + 1) * 60000 for i in range(n)],
        "symbol": ["BTCUSDT"] * 50 + ["ETHUSDT"] * 50,
        "open": [100.0 + i * 0.1 for i in range(n)],
        "high": [101.0 + i * 0.1 for i in range(n)],
        "low": [99.0 + i * 0.1 for i in range(n)],
        "close": [100.5 + i * 0.1 for i in range(n)],
        "volume": [10.0] * n,
    })


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return AggBarMetadata(
        symbols=["BTCUSDT", "ETHUSDT"],
        min_time=1704067200000,
        max_time=1704067200000 + 100 * 60000,
        num_rows=100,
    )


class TestAggBarPolars:
    """Tests for AggBar with Polars DataFrame."""

    def test_create_from_polars_dataframe(self, sample_polars_df, sample_metadata):
        """Test creating AggBar from Polars DataFrame."""
        agg = AggBar(sample_polars_df, sample_metadata)
        
        assert len(agg) == 100
        assert isinstance(agg._data, pl.DataFrame)

    def test_symbols_from_metadata(self, sample_polars_df, sample_metadata):
        """Test that symbols come from metadata, not computed."""
        agg = AggBar(sample_polars_df, sample_metadata)
        
        # Should return metadata symbols directly
        assert agg.symbols == ["BTCUSDT", "ETHUSDT"]

    def test_to_df_returns_pandas(self, sample_polars_df, sample_metadata):
        """Test that to_df() returns Pandas DataFrame."""
        agg = AggBar(sample_polars_df, sample_metadata)
        
        df = agg.to_df()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_getitem_returns_factor(self, sample_polars_df, sample_metadata):
        """Test that __getitem__ returns Factor with correct data."""
        agg = AggBar(sample_polars_df, sample_metadata)
        
        close = agg["close"]
        
        assert close.name == "close"
        assert len(close) == 100

    def test_slice_returns_new_aggbar(self, sample_polars_df, sample_metadata):
        """Test that slice returns new AggBar with filtered data."""
        agg = AggBar(sample_polars_df, sample_metadata)
        
        sliced = agg.slice(symbols=["BTCUSDT"])
        
        assert len(sliced) == 50
        assert sliced.symbols == ["BTCUSDT"]

    def test_cols_property(self, sample_polars_df, sample_metadata):
        """Test cols property returns column names."""
        agg = AggBar(sample_polars_df, sample_metadata)
        
        cols = agg.cols
        
        assert "open" in cols
        assert "close" in cols
        assert "symbol" in cols

    def test_info_returns_pandas_summary(self, sample_polars_df, sample_metadata):
        """Test that info() returns Pandas DataFrame with summary."""
        agg = AggBar(sample_polars_df, sample_metadata)
        
        info = agg.info()
        
        assert isinstance(info, pd.DataFrame)
        assert "num_kbar" in info.columns

    def test_backward_compat_from_pandas(self, sample_metadata):
        """Test backward compatibility: create from Pandas DataFrame."""
        pdf = pd.DataFrame({
            "start_time": [1704067200000, 1704067260000],
            "end_time": [1704067260000, 1704067320000],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "close": [100.0, 101.0],
        })
        
        # Should still work with Pandas input
        agg = AggBar.from_df(pdf)
        
        assert len(agg) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_aggbar_polars.py -v`
Expected: FAIL (AggBar doesn't accept metadata parameter)

**Step 3: Rewrite AggBar to use Polars**

```python
# src/factorium/aggbar.py
"""
Aggregated bar data container for multi-symbol panel data.

AggBar provides a unified interface for working with OHLCV data
across multiple symbols in long format.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Union, List, Optional, TYPE_CHECKING, overload
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    from .factors.core import Factor
    from .data.metadata import AggBarMetadata


class AggBar:
    """
    Multi-symbol bar data container using Polars DataFrame.

    Stores OHLCV data for multiple symbols in long format with columns:
    start_time, end_time, symbol, open, high, low, close, volume, ...

    Args:
        data: Polars DataFrame or Pandas DataFrame with bar data
        metadata: Optional pre-computed metadata (required for Polars input)

    Example:
        >>> from factorium import AggBar
        >>> agg = AggBar(df, metadata)
        >>> close_factor = agg['close']  # Returns Factor
    """

    def __init__(
        self,
        data: Union[pl.DataFrame, pd.DataFrame, List["BaseBar"]],
        metadata: Optional["AggBarMetadata"] = None,
    ):
        # Import here to avoid circular dependency
        from .data.metadata import AggBarMetadata
        
        # Handle different input types
        if isinstance(data, list):
            # Legacy: list of BaseBar objects
            pdf = pd.concat([bar.bars for bar in data])
            self._data = pl.from_pandas(pdf)
            self._metadata = self._compute_metadata()
        elif isinstance(data, pd.DataFrame):
            # Pandas DataFrame input (backward compat)
            self._validate_dataframe_columns(data.columns.tolist())
            if isinstance(data.index, pd.MultiIndex):
                data = data.reset_index()
            self._data = pl.from_pandas(data)
            self._metadata = metadata if metadata else self._compute_metadata()
        elif isinstance(data, pl.DataFrame):
            # Polars DataFrame input (new path)
            self._validate_dataframe_columns(data.columns)
            self._data = data
            if metadata is None:
                self._metadata = self._compute_metadata()
            else:
                self._metadata = metadata
        else:
            raise TypeError(f"Invalid data type: {type(data)}")

        # Sort by end_time, symbol
        self._data = self._data.sort(["end_time", "symbol"])

    def _validate_dataframe_columns(self, columns: list) -> None:
        """Validate required columns exist."""
        required = {"start_time", "end_time", "symbol"}
        missing = required - set(columns)
        if missing:
            raise ValueError(f"DataFrame must contain columns: {missing}")

    def _compute_metadata(self) -> "AggBarMetadata":
        """Compute metadata from actual data."""
        from .data.metadata import AggBarMetadata
        
        if self._data.is_empty():
            return AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)
        
        symbols = self._data["symbol"].unique().sort().to_list()
        min_time = self._data["start_time"].min()
        max_time = self._data["end_time"].max()
        num_rows = len(self._data)
        
        return AggBarMetadata(
            symbols=symbols,
            min_time=min_time,
            max_time=max_time,
            num_rows=num_rows,
        )

    @classmethod
    def from_bars(cls, bars: List["BaseBar"]) -> "AggBar":
        """Create AggBar from a list of BaseBar objects."""
        return cls(bars)

    @classmethod
    def from_df(cls, df: Union[pd.DataFrame, pl.DataFrame]) -> "AggBar":
        """Create AggBar from a DataFrame."""
        return cls(df)

    @classmethod
    def from_csv(cls, path: Path) -> "AggBar":
        """Create AggBar from a CSV file."""
        df = pl.read_csv(path)
        return cls(df)

    def to_df(self) -> pd.DataFrame:
        """Return the data as a Pandas DataFrame copy."""
        return self._data.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        """Return the data as a Polars DataFrame copy."""
        return self._data.clone()

    def to_csv(self, path: Path) -> Path:
        """Save data to a CSV file."""
        self._data.write_csv(path)
        return path

    def to_parquet(self, path: Path) -> Path:
        """Save data to a Parquet file."""
        self._data.write_parquet(path)
        return path

    def __getitem__(self, key: Union[str, List[str]]) -> Union["Factor", "AggBar"]:
        """
        Get a column as a Factor or multiple columns as a new AggBar.

        Args:
            key: Column name (str) or list of column names

        Returns:
            Factor if single column, AggBar if multiple columns
        """
        if isinstance(key, str):
            if key not in self._data.columns:
                raise KeyError(f"Column {key} not found in the dataframe")

            # Lazy import to avoid circular dependency
            from .factors.core import Factor

            factor_df = self._data.select(["start_time", "end_time", "symbol", key])
            factor_df = factor_df.rename({key: "factor"})
            return Factor(factor_df, name=key)

        elif isinstance(key, list):
            cols = ["start_time", "end_time", "symbol"] + [
                c for c in key if c not in ["start_time", "end_time", "symbol"]
            ]
            new_data = self._data.select(cols)
            # Recompute metadata for subset
            return AggBar(new_data)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def slice(
        self,
        start: Optional[Union[datetime, int, str]] = None,
        end: Optional[Union[datetime, int, str]] = None,
        symbols: Optional[List[str]] = None,
    ) -> "AggBar":
        """
        Slice data by time range and/or symbols.

        Args:
            start: Start time (datetime, timestamp, or 'YYYY-MM-DD HH:MM:SS')
            end: End time (datetime, timestamp, or 'YYYY-MM-DD HH:MM:SS')
            symbols: List of symbols to include

        Returns:
            New AggBar with filtered data
        """

        def convert_timestamp(value: Optional[Union[datetime, int, str]]) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, str):
                dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S") if " " in value else datetime.strptime(value, "%Y-%m-%d")
                return int(dt.timestamp() * 1000)
            elif isinstance(value, int):
                if len(str(value)) == 13:
                    return value
                elif len(str(value)) == 10:
                    return value * 1000
                else:
                    raise ValueError(f"Invalid timestamp: {value}")
            elif isinstance(value, datetime):
                return int(value.timestamp() * 1000)
            else:
                raise ValueError(f"Cannot convert {value} to timestamp")

        start_ts = convert_timestamp(start)
        end_ts = convert_timestamp(end)

        if symbols is None:
            symbols = self.symbols

        # Build filter expression
        expr = pl.col("symbol").is_in(symbols)
        if start_ts is not None:
            expr = expr & (pl.col("start_time") >= start_ts)
        if end_ts is not None:
            expr = expr & (pl.col("end_time") <= end_ts)

        filtered = self._data.filter(expr)
        return AggBar(filtered)

    @property
    def cols(self) -> List[str]:
        """Return list of column names."""
        return self._data.columns

    @property
    def symbols(self) -> List[str]:
        """Return list of unique symbols from metadata."""
        return self._metadata.symbols

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Return unique timestamps from start_time and end_time."""
        ts1 = self._data["start_time"].unique().to_numpy()
        ts2 = self._data["end_time"].unique().to_numpy()
        all_ts = np.unique(np.concatenate([ts1, ts2]))
        return pd.DatetimeIndex(pd.to_datetime(all_ts, unit="ms"))

    @property
    def metadata(self) -> "AggBarMetadata":
        """Return the metadata object."""
        return self._metadata

    def info(self) -> pd.DataFrame:
        """
        Get summary information for each symbol.

        Returns:
            DataFrame with num_kbar, start_time, end_time, num_nan per symbol
        """
        # Use Polars for aggregation, convert to Pandas for output
        info_df = (
            self._data
            .group_by("symbol")
            .agg([
                pl.len().alias("num_kbar"),
                pl.col("start_time").min().alias("start_time"),
                pl.col("end_time").max().alias("end_time"),
                pl.all().null_count().sum().alias("num_nan"),
            ])
        ).to_pandas()
        
        # Convert timestamps to datetime
        info_df["start_time"] = pd.to_datetime(info_df["start_time"], unit="ms")
        info_df["end_time"] = pd.to_datetime(info_df["end_time"], unit="ms")
        
        return info_df.set_index("symbol")

    def copy(self) -> "AggBar":
        """Return a copy of this AggBar."""
        return AggBar(self._data.clone(), self._metadata)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self):
        n_symbols = len(self.symbols)
        ts = self.timestamps
        if len(ts) > 0:
            time_range = f"{ts.min().strftime('%Y-%m-%d %H:%M:%S')} - {ts.max().strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            time_range = "N/A"

        return f"AggBar: {len(self)} rows, {len(self.cols)} columns, symbols={n_symbols}, time_range={time_range}"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_aggbar_polars.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add src/factorium/aggbar.py tests/test_aggbar_polars.py
git commit -m "refactor(aggbar): migrate from Pandas to Polars DataFrame storage"
```

---

## Task 4: Update BinanceDataLoader to use new aggregator output

**Files:**
- Modify: `src/factorium/data/loader.py`
- Modify: `tests/test_data_loader.py`

**Step 1: Modify load_aggbar to use Polars**

Key changes in `load_aggbar`:
1. Change `all_dfs: List[pd.DataFrame]` to `all_dfs: List[pl.DataFrame]`
2. Collect metadata from each aggregation
3. Use `pl.concat()` instead of `pd.concat()`
4. Pass metadata to AggBar constructor

```python
# In load_aggbar method:

all_dfs: List[pl.DataFrame] = []
all_metadata: List[AggBarMetadata] = []

# ... in the aggregation loop:
df, meta = aggregator.aggregate_time_bars(...)
if not df.is_empty():
    all_dfs.append(df)
    all_metadata.append(meta)

# ... at the end:
result_df = pl.concat(all_dfs)
combined_meta = AggBarMetadata.merge(all_metadata)

return AggBar(result_df, combined_meta)
```

**Step 2: Update imports**

```python
import polars as pl
from .metadata import AggBarMetadata
```

**Step 3: Run tests**

Run: `pytest tests/test_data_loader.py -v`
Expected: Some failures due to changed return types

**Step 4: Update tests as needed**

Tests that check for `pd.DataFrame` should be updated or marked as deprecated.

**Step 5: Commit**

```bash
git add src/factorium/data/loader.py tests/test_data_loader.py
git commit -m "refactor(loader): use Polars DataFrame and metadata from aggregator"
```

---

## Task 5: Update BarCache to use Polars

**Files:**
- Modify: `src/factorium/data/cache.py`
- Test: `tests/data/test_cache_polars.py`

**Step 1: Write the failing test**

```python
# tests/data/test_cache_polars.py
"""Tests for BarCache with Polars support."""

import pytest
import polars as pl
import tempfile
from pathlib import Path
from datetime import datetime

from factorium.data.cache import BarCache


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_polars_df():
    """Create sample Polars DataFrame."""
    return pl.DataFrame({
        "start_time": [1704067200000, 1704067260000],
        "end_time": [1704067260000, 1704067320000],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [10.0, 20.0],
    })


class TestBarCachePolars:
    """Tests for BarCache with Polars DataFrame."""

    def test_put_and_get_polars(self, temp_cache_dir, sample_polars_df):
        """Test storing and retrieving Polars DataFrame."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        cache.put(
            df=sample_polars_df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures_um",
            date=datetime(2024, 1, 1),
        )
        
        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures_um",
            date=datetime(2024, 1, 1),
        )
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

    def test_get_returns_none_when_not_cached(self, temp_cache_dir):
        """Test that get returns None when data not in cache."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures_um",
            date=datetime(2024, 1, 1),
        )
        
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_cache_polars.py -v`
Expected: FAIL (returns pd.DataFrame, not pl.DataFrame)

**Step 3: Update BarCache to use Polars**

```python
# src/factorium/data/cache.py
"""Daily cache layer for pre-aggregated bar data."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


class BarCache:
    # ... (keep existing methods)

    def get(
        self,
        exchange: str,
        symbols: list[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        date: datetime,
    ) -> pl.DataFrame | None:
        """Get cached data for a single date.

        Returns:
            Polars DataFrame if cached, None otherwise
        """
        cache_path = self._get_cache_path(exchange, symbols, interval_ms, data_type, market_type, date)

        if cache_path.exists():
            return pl.read_parquet(cache_path)
        return None

    def get_range(
        self,
        exchange: str,
        symbols: list[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame | None:
        """Get cached data for a date range.

        Returns None if any date in the range is missing from cache.
        """
        dfs = []
        current = start_date

        while current < end_date:
            df = self.get(exchange, symbols, interval_ms, data_type, market_type, current)
            if df is None:
                return None
            dfs.append(df)
            current += timedelta(days=1)

        if not dfs:
            return None

        return pl.concat(dfs)

    def put(
        self,
        df: pl.DataFrame,
        exchange: str,
        symbols: list[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        date: datetime,
    ) -> Path:
        """Store data in cache for a single date.

        Returns:
            Path to cached file
        """
        cache_path = self._get_cache_path(exchange, symbols, interval_ms, data_type, market_type, date)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
        return cache_path
```

**Step 4: Run tests**

Run: `pytest tests/data/test_cache_polars.py -v`
Expected: All passed

**Step 5: Commit**

```bash
git add src/factorium/data/cache.py tests/data/test_cache_polars.py
git commit -m "refactor(cache): migrate from Pandas to Polars for parquet storage"
```

---

## Task 6: Update Factor to accept Polars from AggBar directly

**Files:**
- Modify: `src/factorium/factors/base.py:26-48`

**Step 1: Verify Factor already handles Polars**

The `_to_lazy` method already handles `pl.DataFrame`:
```python
if isinstance(data, pl.DataFrame):
    return self._normalize_schema_lazy(data.lazy())
```

**Step 2: Update AggBar check to use Polars path**

In `_to_lazy`, the `hasattr(data, "to_df")` check currently calls `to_df()` which returns Pandas.
Update to use the Polars path:

```python
def _to_lazy(self, data: Union["AggBar", pd.DataFrame, pl.DataFrame, pl.LazyFrame, Path]) -> pl.LazyFrame:
    # ... existing Path handling ...

    if isinstance(data, pl.LazyFrame):
        return self._normalize_schema_lazy(data)

    if isinstance(data, pl.DataFrame):
        return self._normalize_schema_lazy(data.lazy())

    # Check for AggBar (has to_polars method)
    if hasattr(data, "to_polars"):
        return self._normalize_schema_lazy(data.to_polars().lazy())
    
    # Legacy: AggBar with only to_df (Pandas)
    if hasattr(data, "to_df"):
        return self._normalize_schema_pandas(data.to_df())

    if isinstance(data, pd.DataFrame):
        return self._normalize_schema_pandas(data)

    raise ValueError(f"Invalid data type: {type(data)}")
```

**Step 3: Run Factor tests**

Run: `pytest tests/factors/ -v`
Expected: All passed

**Step 4: Commit**

```bash
git add src/factorium/factors/base.py
git commit -m "refactor(factor): prefer Polars path when loading from AggBar"
```

---

## Task 7: Update existing tests for new API

**Files:**
- Modify: `tests/data/test_aggregator_bars.py`
- Modify: `tests/test_data_loader.py`

**Step 1: Update aggregator bar tests**

Tests currently expect `pd.DataFrame`, need to handle tuple return:

```python
# Before:
df = aggregator.aggregate_time_bars(...)
assert isinstance(df, pd.DataFrame)

# After:
df, meta = aggregator.aggregate_time_bars(...)
assert isinstance(df, pl.DataFrame)
assert isinstance(meta, AggBarMetadata)
```

**Step 2: Update loader tests**

Update tests that check `agg.data` (now Polars):

```python
# Before:
btc_data = agg.data[agg.data["symbol"] == "BTCUSDT"]

# After:
btc_data = agg._data.filter(pl.col("symbol") == "BTCUSDT")
# Or use to_df() for test convenience:
btc_data = agg.to_df().query("symbol == 'BTCUSDT'")
```

**Step 3: Run full test suite**

Run: `pytest -v`
Expected: All passed

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: update tests for Polars migration"
```

---

## Task 8: Remove deprecated load_data if unused

**Files:**
- Modify: `src/factorium/data/loader.py`

**Step 1: Check if load_data is used anywhere**

Search for `load_data` usage in the codebase.

**Step 2: If unused, deprecate or remove**

If `load_data` is not used externally, add deprecation warning:

```python
def load_data(self, ...):
    """Deprecated: Use load_aggbar instead."""
    import warnings
    warnings.warn(
        "load_data is deprecated, use load_aggbar instead",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing code ...
```

**Step 3: Commit**

```bash
git add src/factorium/data/loader.py
git commit -m "deprecate: mark load_data as deprecated in favor of load_aggbar"
```

---

## Task 9: Final verification and cleanup

**Step 1: Run full test suite**

Run: `pytest -v --tb=short`
Expected: All tests pass

**Step 2: Run type checking (if available)**

Run: `mypy src/factorium/` or `pyright`

**Step 3: Update exports in __init__.py if needed**

Ensure `AggBarMetadata` is exported if needed externally.

**Step 4: Final commit**

```bash
git add .
git commit -m "chore: final cleanup for Polars migration"
```

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 0 | Add Polars import to aggregator | aggregator.py |
| 1 | Create AggBarMetadata dataclass | metadata.py, test_metadata.py |
| 2 | Modify aggregator to return Polars + metadata | aggregator.py, test_aggregator_polars.py |
| 3 | Migrate AggBar to Polars | aggbar.py, test_aggbar_polars.py |
| 4 | Update loader to use new aggregator | loader.py, test_data_loader.py |
| 5 | Update cache to use Polars | cache.py, test_cache_polars.py |
| 6 | Update Factor to prefer Polars path | base.py |
| 7 | Update existing tests | various test files |
| 8 | Deprecate load_data | loader.py |
| 9 | Final verification | - |

**Breaking Changes:**
- `AggBar.data` is now `pl.DataFrame` (was `pd.DataFrame`)
- `BarAggregator.aggregate_*_bars()` now returns `tuple[pl.DataFrame, AggBarMetadata]`
- `BarCache.get()` returns `pl.DataFrame | None`

**Migration for users:**
- Use `agg.to_df()` to get Pandas DataFrame
- Use `agg._data` for direct Polars access
- Use `agg.metadata` for pre-computed statistics
