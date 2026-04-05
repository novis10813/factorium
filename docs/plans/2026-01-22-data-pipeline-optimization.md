# Data Pipeline Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize data loading and factor computation to support 100+ symbols efficiently with 16GB RAM, achieving 10x speedup via DuckDB SQL aggregation, Polars computation engine, and daily caching.

**Architecture:**
- Data Loading: DuckDB SQL aggregation (no tick data in Python memory)
- Factor Computation: Polars-based engine with lazy Pandas conversion
- Caching: Daily Parquet files for pre-aggregated bars
- Exchange Abstraction: Adapter pattern for multi-exchange support

**Tech Stack:** DuckDB, Polars, Pandas (user API), PyArrow

---

## Phase 1: Data Loading Optimization

### Task 1: Add Polars Dependency

**Files:**
- Modify: `pyproject.toml:24-35`

**Step 1: Add polars to dependencies**

```toml
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "numba>=0.57.0",
    "aiohttp>=3.9.0",
    "aiofiles>=23.0.0",
    "matplotlib>=3.7.0",
    "pyparsing>=3.0.0",
    "duckdb>=1.0.0",
    "pyarrow>=14.0.0",
    "polars>=1.0.0",
]
```

**Step 2: Install dependencies**

Run: `uv sync`
Expected: Success, polars installed

**Step 3: Verify installation**

Run: `uv run python -c "import polars; print(polars.__version__)"`
Expected: Version number printed (e.g., "1.x.x")

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add polars for high-performance data processing"
```

---

### Task 2: Create Exchange Adapter Base Class

**Files:**
- Create: `src/factorium/data/adapters/__init__.py`
- Create: `src/factorium/data/adapters/base.py`
- Test: `tests/data/test_adapters.py`

**Step 1: Create adapters directory**

Run: `mkdir -p src/factorium/data/adapters tests/data`

**Step 2: Write the failing test**

Create `tests/data/__init__.py`:
```python
# Tests for data module
```

Create `tests/data/test_adapters.py`:
```python
"""Tests for exchange adapters."""

import pytest
from factorium.data.adapters.base import BaseExchangeAdapter, ColumnMapping


class TestColumnMapping:
    """Tests for ColumnMapping dataclass."""

    def test_column_mapping_creation(self):
        """Test ColumnMapping can be created with required fields."""
        mapping = ColumnMapping(
            timestamp="transact_time",
            price="price",
            volume="quantity",
            is_buyer_maker="is_buyer_maker",
        )
        
        assert mapping.timestamp == "transact_time"
        assert mapping.price == "price"
        assert mapping.volume == "quantity"
        assert mapping.is_buyer_maker == "is_buyer_maker"


class TestBaseExchangeAdapter:
    """Tests for BaseExchangeAdapter abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseExchangeAdapter cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseExchangeAdapter()

    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclass must implement all abstract methods."""
        
        class IncompleteAdapter(BaseExchangeAdapter):
            pass
        
        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_complete_subclass_can_be_instantiated(self):
        """Test that a complete subclass can be instantiated."""
        
        class CompleteAdapter(BaseExchangeAdapter):
            @property
            def name(self) -> str:
                return "test"
            
            @property
            def column_mappings(self) -> dict:
                return {
                    "aggTrades": ColumnMapping(
                        timestamp="ts",
                        price="p",
                        volume="v",
                        is_buyer_maker="m",
                    )
                }
            
            def build_parquet_glob(self, *args, **kwargs) -> str:
                return "test/*.parquet"
            
            def get_download_url(self, *args, **kwargs) -> str:
                return "https://example.com"
        
        adapter = CompleteAdapter()
        assert adapter.name == "test"
        assert "aggTrades" in adapter.column_mappings
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/data/test_adapters.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'factorium.data.adapters'"

**Step 4: Create adapters/__init__.py**

```python
"""Exchange adapters for multi-exchange support."""

from .base import BaseExchangeAdapter, ColumnMapping

__all__ = ["BaseExchangeAdapter", "ColumnMapping"]
```

**Step 5: Create base.py with implementation**

```python
"""Base class for exchange adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ColumnMapping:
    """Column name mapping for exchange-specific data formats.
    
    Attributes:
        timestamp: Column name for transaction timestamp (milliseconds)
        price: Column name for trade price
        volume: Column name for trade volume/quantity
        is_buyer_maker: Column name for buyer/seller indicator
    """
    timestamp: str
    price: str
    volume: str
    is_buyer_maker: str


class BaseExchangeAdapter(ABC):
    """Abstract base class for exchange data adapters.
    
    Provides a unified interface for:
    - Column name mapping across different exchanges
    - Parquet file path construction
    - Download URL generation
    
    Subclasses must implement all abstract methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return exchange name (e.g., 'binance', 'bybit')."""
        ...
    
    @property
    @abstractmethod
    def column_mappings(self) -> Dict[str, ColumnMapping]:
        """Return column mappings for each data type.
        
        Returns:
            Dict mapping data_type (e.g., 'aggTrades') to ColumnMapping
        """
        ...
    
    @abstractmethod
    def build_parquet_glob(
        self,
        base_path: Path,
        symbols: List[str],
        data_type: str,
        market_type: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> str:
        """Build glob pattern for Parquet files.
        
        Args:
            base_path: Base directory for data storage
            symbols: List of trading symbols
            data_type: Type of data (e.g., 'aggTrades', 'trades')
            market_type: Market type (e.g., 'futures', 'spot')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Exchange-specific parameters
            
        Returns:
            Glob pattern string for matching Parquet files
        """
        ...
    
    @abstractmethod
    def get_download_url(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        date: str,
        **kwargs,
    ) -> str:
        """Get download URL for a specific data file.
        
        Args:
            symbol: Trading symbol
            data_type: Type of data
            market_type: Market type
            date: Date string (YYYY-MM-DD)
            **kwargs: Exchange-specific parameters
            
        Returns:
            Download URL string
        """
        ...
    
    def get_column_mapping(self, data_type: str) -> ColumnMapping:
        """Get column mapping for a specific data type.
        
        Args:
            data_type: Type of data (e.g., 'aggTrades')
            
        Returns:
            ColumnMapping for the data type
            
        Raises:
            KeyError: If data_type is not supported
        """
        if data_type not in self.column_mappings:
            raise KeyError(
                f"Unsupported data type: {data_type}. "
                f"Available: {list(self.column_mappings.keys())}"
            )
        return self.column_mappings[data_type]
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/data/test_adapters.py -v`
Expected: PASS (3 tests)

**Step 7: Commit**

```bash
git add src/factorium/data/adapters/ tests/data/
git commit -m "feat(data): add BaseExchangeAdapter abstract class for multi-exchange support"
```

---

### Task 3: Create Binance Adapter

**Files:**
- Create: `src/factorium/data/adapters/binance.py`
- Modify: `src/factorium/data/adapters/__init__.py`
- Test: `tests/data/test_adapters.py` (append)

**Step 1: Write the failing test**

Append to `tests/data/test_adapters.py`:
```python
from pathlib import Path
from factorium.data.adapters.binance import BinanceAdapter


class TestBinanceAdapter:
    """Tests for BinanceAdapter."""

    @pytest.fixture
    def adapter(self):
        return BinanceAdapter()

    def test_name(self, adapter):
        """Test adapter name is 'binance'."""
        assert adapter.name == "binance"

    def test_column_mappings_aggtrades(self, adapter):
        """Test column mapping for aggTrades."""
        mapping = adapter.get_column_mapping("aggTrades")
        
        assert mapping.timestamp == "transact_time"
        assert mapping.price == "price"
        assert mapping.volume == "quantity"
        assert mapping.is_buyer_maker == "is_buyer_maker"

    def test_column_mappings_trades(self, adapter):
        """Test column mapping for trades."""
        mapping = adapter.get_column_mapping("trades")
        
        assert mapping.timestamp == "time"
        assert mapping.price == "price"
        assert mapping.volume == "qty"
        assert mapping.is_buyer_maker == "is_buyer_maker"

    def test_build_parquet_glob_futures_um(self, adapter):
        """Test glob pattern for futures UM market."""
        pattern = adapter.build_parquet_glob(
            base_path=Path("/data"),
            symbols=["BTCUSDT"],
            data_type="aggTrades",
            market_type="futures",
            start_date="2024-01-01",
            end_date="2024-01-07",
            futures_type="um",
        )
        
        assert "market=futures_um" in pattern
        assert "data_type=aggTrades" in pattern
        assert "symbol=BTCUSDT" in pattern
        assert "**/*.parquet" in pattern

    def test_build_parquet_glob_spot(self, adapter):
        """Test glob pattern for spot market."""
        pattern = adapter.build_parquet_glob(
            base_path=Path("/data"),
            symbols=["BTCUSDT"],
            data_type="trades",
            market_type="spot",
            start_date="2024-01-01",
            end_date="2024-01-07",
        )
        
        assert "market=spot" in pattern
        assert "data_type=trades" in pattern

    def test_build_parquet_glob_multiple_symbols(self, adapter):
        """Test glob pattern for multiple symbols uses wildcard."""
        pattern = adapter.build_parquet_glob(
            base_path=Path("/data"),
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            data_type="aggTrades",
            market_type="futures",
            start_date="2024-01-01",
            end_date="2024-01-07",
            futures_type="um",
        )
        
        # Multiple symbols should use wildcard or union pattern
        # Implementation can choose either approach
        assert "symbol=" in pattern

    def test_get_download_url_futures_um(self, adapter):
        """Test download URL for futures UM."""
        url = adapter.get_download_url(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            date="2024-01-15",
            futures_type="um",
        )
        
        assert "data.binance.vision" in url
        assert "futures/um" in url
        assert "BTCUSDT" in url
        assert "aggTrades" in url
        assert "2024-01-15" in url
        assert ".zip" in url

    def test_unsupported_data_type_raises(self, adapter):
        """Test that unsupported data type raises KeyError."""
        with pytest.raises(KeyError, match="Unsupported data type"):
            adapter.get_column_mapping("invalid_type")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_adapters.py::TestBinanceAdapter -v`
Expected: FAIL with "ModuleNotFoundError" or import error

**Step 3: Create binance.py implementation**

```python
"""Binance exchange adapter."""

from pathlib import Path
from typing import Dict, List

from .base import BaseExchangeAdapter, ColumnMapping


class BinanceAdapter(BaseExchangeAdapter):
    """Adapter for Binance exchange data.
    
    Supports:
    - Spot market: trades, aggTrades, klines
    - Futures UM (USDT-margined): trades, aggTrades, klines
    - Futures CM (Coin-margined): trades, aggTrades, klines
    """
    
    BINANCE_VISION_BASE = "https://data.binance.vision/data"
    
    @property
    def name(self) -> str:
        return "binance"
    
    @property
    def column_mappings(self) -> Dict[str, ColumnMapping]:
        return {
            "aggTrades": ColumnMapping(
                timestamp="transact_time",
                price="price",
                volume="quantity",
                is_buyer_maker="is_buyer_maker",
            ),
            "trades": ColumnMapping(
                timestamp="time",
                price="price",
                volume="qty",
                is_buyer_maker="is_buyer_maker",
            ),
            "klines": ColumnMapping(
                timestamp="open_time",
                price="close",
                volume="volume",
                is_buyer_maker="",  # Not applicable for klines
            ),
        }
    
    def _get_market_string(self, market_type: str, futures_type: str = "") -> str:
        """Get market string for Hive partitioning."""
        if market_type == "futures":
            return f"futures_{futures_type}"
        return market_type
    
    def build_parquet_glob(
        self,
        base_path: Path,
        symbols: List[str],
        data_type: str,
        market_type: str,
        start_date: str,
        end_date: str,
        futures_type: str = "um",
        **kwargs,
    ) -> str:
        """Build glob pattern for Binance Parquet files.
        
        Uses Hive partitioning format:
        base_path/market=X/data_type=Y/symbol=Z/year=.../month=.../day=.../data.parquet
        """
        market_str = self._get_market_string(market_type, futures_type)
        
        if len(symbols) == 1:
            symbol_pattern = f"symbol={symbols[0]}"
        else:
            # For multiple symbols, we'll filter in SQL instead
            symbol_pattern = "symbol=*"
        
        return str(
            base_path
            / f"market={market_str}"
            / f"data_type={data_type}"
            / symbol_pattern
            / "**/*.parquet"
        )
    
    def get_download_url(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        date: str,
        futures_type: str = "um",
        **kwargs,
    ) -> str:
        """Get Binance Vision download URL."""
        if market_type == "futures":
            market_path = f"futures/{futures_type}"
        else:
            market_path = "spot"
        
        return (
            f"{self.BINANCE_VISION_BASE}/{market_path}/daily/{data_type}/"
            f"{symbol}/{symbol}-{data_type}-{date}.zip"
        )
```

**Step 4: Update adapters/__init__.py**

```python
"""Exchange adapters for multi-exchange support."""

from .base import BaseExchangeAdapter, ColumnMapping
from .binance import BinanceAdapter

__all__ = ["BaseExchangeAdapter", "ColumnMapping", "BinanceAdapter"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/data/test_adapters.py -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/factorium/data/adapters/
git commit -m "feat(data): add BinanceAdapter for Binance exchange"
```

---

### Task 4: Create Bar Aggregator (DuckDB SQL)

**Files:**
- Create: `src/factorium/data/aggregator.py`
- Test: `tests/data/test_aggregator.py`

**Step 1: Write the failing test**

Create `tests/data/test_aggregator.py`:
```python
"""Tests for BarAggregator."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

from factorium.data.aggregator import BarAggregator
from factorium.data.adapters import ColumnMapping


@pytest.fixture
def sample_parquet_dir():
    """Create temporary directory with sample Parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create Hive-partitioned structure
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for day in [1, 2, 3]:
                partition_path = (
                    tmpdir
                    / "market=futures_um"
                    / "data_type=aggTrades"
                    / f"symbol={symbol}"
                    / "year=2024"
                    / "month=01"
                    / f"day={day:02d}"
                )
                partition_path.mkdir(parents=True, exist_ok=True)
                
                # Generate sample trade data
                np.random.seed(42 + day + hash(symbol) % 100)
                n_trades = 1000
                base_ts = int(datetime(2024, 1, day).timestamp() * 1000)
                
                df = pd.DataFrame({
                    "transact_time": base_ts + np.arange(n_trades) * 60000,  # 1 trade per minute
                    "price": 100.0 + np.cumsum(np.random.randn(n_trades) * 0.1),
                    "quantity": np.abs(np.random.randn(n_trades)) * 10 + 1,
                    "is_buyer_maker": np.random.choice([True, False], n_trades),
                })
                
                table = pa.Table.from_pandas(df)
                pq.write_table(table, partition_path / "data.parquet")
        
        yield tmpdir


@pytest.fixture
def column_mapping():
    return ColumnMapping(
        timestamp="transact_time",
        price="price",
        volume="quantity",
        is_buyer_maker="is_buyer_maker",
    )


class TestBarAggregator:
    """Tests for BarAggregator."""

    def test_aggregate_time_bars_single_symbol(self, sample_parquet_dir, column_mapping):
        """Test aggregation for a single symbol."""
        aggregator = BarAggregator()
        
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")
        
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 4).timestamp() * 1000)
        
        result = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,  # 1 minute
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Check required columns
        required_cols = ["symbol", "start_time", "end_time", "open", "high", "low", "close", "volume", "vwap"]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"
        
        # Check symbol is correct
        assert result["symbol"].unique().tolist() == ["BTCUSDT"]

    def test_aggregate_time_bars_multiple_symbols(self, sample_parquet_dir, column_mapping):
        """Test aggregation for multiple symbols."""
        aggregator = BarAggregator()
        
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=*/**/*.parquet")
        
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 4).timestamp() * 1000)
        
        result = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT", "ETHUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )
        
        assert len(result) > 0
        assert set(result["symbol"].unique()) == {"BTCUSDT", "ETHUSDT"}

    def test_aggregate_time_bars_different_intervals(self, sample_parquet_dir, column_mapping):
        """Test that different intervals produce different bar counts."""
        aggregator = BarAggregator()
        
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")
        
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2).timestamp() * 1000)
        
        result_1m = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,  # 1 minute
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )
        
        result_5m = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=300_000,  # 5 minutes
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )
        
        # 5-minute bars should have ~5x fewer rows than 1-minute bars
        assert len(result_1m) > len(result_5m)
        assert len(result_1m) >= len(result_5m) * 4  # At least 4x

    def test_aggregate_includes_buyer_seller_stats(self, sample_parquet_dir, column_mapping):
        """Test that buyer/seller statistics are included."""
        aggregator = BarAggregator()
        
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")
        
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2).timestamp() * 1000)
        
        result = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
            include_buyer_seller=True,
        )
        
        buyer_seller_cols = ["num_buyer", "num_seller", "num_buyer_volume", "num_seller_volume"]
        for col in buyer_seller_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_aggregate_without_buyer_seller_stats(self, sample_parquet_dir, column_mapping):
        """Test aggregation without buyer/seller statistics."""
        aggregator = BarAggregator()
        
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")
        
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2).timestamp() * 1000)
        
        result = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
            include_buyer_seller=False,
        )
        
        # Should not have buyer/seller columns
        assert "num_buyer" not in result.columns

    def test_vwap_calculation_correct(self, sample_parquet_dir, column_mapping):
        """Test that VWAP is correctly calculated."""
        aggregator = BarAggregator()
        
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")
        
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2).timestamp() * 1000)
        
        result = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )
        
        # VWAP should be between low and high
        for _, row in result.iterrows():
            assert row["low"] <= row["vwap"] <= row["high"], \
                f"VWAP {row['vwap']} not between low {row['low']} and high {row['high']}"

    def test_ohlc_integrity(self, sample_parquet_dir, column_mapping):
        """Test OHLC data integrity (high >= low, etc.)."""
        aggregator = BarAggregator()
        
        pattern = str(sample_parquet_dir / "market=futures_um/data_type=aggTrades/symbol=BTCUSDT/**/*.parquet")
        
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 2).timestamp() * 1000)
        
        result = aggregator.aggregate_time_bars(
            parquet_pattern=pattern,
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_ts=start_ts,
            end_ts=end_ts,
            column_mapping=column_mapping,
        )
        
        # High >= Low
        assert (result["high"] >= result["low"]).all()
        
        # High >= Open and High >= Close
        assert (result["high"] >= result["open"]).all()
        assert (result["high"] >= result["close"]).all()
        
        # Low <= Open and Low <= Close
        assert (result["low"] <= result["open"]).all()
        assert (result["low"] <= result["close"]).all()
        
        # Volume > 0
        assert (result["volume"] > 0).all()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_aggregator.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create aggregator.py implementation**

```python
"""DuckDB-based bar aggregator for high-performance OHLCV aggregation."""

import duckdb
import pandas as pd
from typing import List

from .adapters.base import ColumnMapping


class BarAggregator:
    """High-performance bar aggregator using DuckDB SQL.
    
    Aggregates tick/trade data into OHLCV bars directly in DuckDB,
    avoiding loading raw tick data into Python memory.
    
    Benefits:
    - Memory efficient: aggregation happens in DuckDB
    - Fast: uses DuckDB's vectorized execution engine
    - Scalable: handles large datasets that don't fit in memory
    """
    
    def aggregate_time_bars(
        self,
        parquet_pattern: str,
        symbols: List[str],
        interval_ms: int,
        start_ts: int,
        end_ts: int,
        column_mapping: ColumnMapping,
        include_buyer_seller: bool = True,
    ) -> pd.DataFrame:
        """Aggregate tick data into time-based OHLCV bars.
        
        Args:
            parquet_pattern: Glob pattern for Parquet files
            symbols: List of symbols to include
            interval_ms: Bar interval in milliseconds
            start_ts: Start timestamp (milliseconds)
            end_ts: End timestamp (milliseconds)
            column_mapping: Column name mapping for the data source
            include_buyer_seller: Include buyer/seller statistics
            
        Returns:
            DataFrame with columns:
            - symbol, start_time, end_time
            - open, high, low, close, volume, vwap
            - (optional) num_buyer, num_seller, num_buyer_volume, num_seller_volume
        """
        ts_col = column_mapping.timestamp
        price_col = column_mapping.price
        volume_col = column_mapping.volume
        ibm_col = column_mapping.is_buyer_maker
        
        # Build symbol filter
        symbols_str = ", ".join([f"'{s}'" for s in symbols])
        
        # Build buyer/seller aggregation SQL
        buyer_seller_sql = ""
        buyer_seller_cols = ""
        if include_buyer_seller and ibm_col:
            buyer_seller_sql = f"""
                , SUM(CASE WHEN NOT {ibm_col} THEN 1 ELSE 0 END) AS num_buyer
                , SUM(CASE WHEN {ibm_col} THEN 1 ELSE 0 END) AS num_seller
                , SUM(CASE WHEN NOT {ibm_col} THEN {volume_col} ELSE 0 END) AS num_buyer_volume
                , SUM(CASE WHEN {ibm_col} THEN {volume_col} ELSE 0 END) AS num_seller_volume
            """
            buyer_seller_cols = ", num_buyer, num_seller, num_buyer_volume, num_seller_volume"
        
        query = f"""
            WITH raw_data AS (
                SELECT 
                    symbol,
                    {ts_col} AS ts,
                    {price_col} AS price,
                    {volume_col} AS volume
                    {f', {ibm_col} AS is_buyer_maker' if include_buyer_seller and ibm_col else ''}
                FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
                WHERE symbol IN ({symbols_str})
                  AND {ts_col} >= {start_ts}
                  AND {ts_col} < {end_ts}
            ),
            aggregated AS (
                SELECT 
                    symbol,
                    (ts // {interval_ms}) * {interval_ms} AS start_time,
                    (ts // {interval_ms}) * {interval_ms} + {interval_ms} AS end_time,
                    FIRST(price) AS open,
                    MAX(price) AS high,
                    MIN(price) AS low,
                    LAST(price) AS close,
                    SUM(volume) AS volume,
                    SUM(price * volume) / SUM(volume) AS vwap
                    {buyer_seller_sql}
                FROM raw_data
                GROUP BY symbol, (ts // {interval_ms})
            )
            SELECT 
                symbol, start_time, end_time,
                open, high, low, close, volume, vwap
                {buyer_seller_cols}
            FROM aggregated
            ORDER BY start_time, symbol
        """
        
        return duckdb.query(query).df()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_aggregator.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/factorium/data/aggregator.py tests/data/test_aggregator.py
git commit -m "feat(data): add BarAggregator for DuckDB SQL-based OHLCV aggregation"
```

---

### Task 5: Create Bar Cache

**Files:**
- Create: `src/factorium/data/cache.py`
- Test: `tests/data/test_cache.py`

**Step 1: Write the failing test**

Create `tests/data/test_cache.py`:
```python
"""Tests for BarCache."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from factorium.data.cache import BarCache


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_bar_df():
    """Create sample bar DataFrame."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1min")
    
    return pd.DataFrame({
        "symbol": ["BTCUSDT"] * 50 + ["ETHUSDT"] * 50,
        "start_time": np.tile(dates[:50].astype(np.int64) // 10**6, 2),
        "end_time": np.tile((dates[:50] + pd.Timedelta(minutes=1)).astype(np.int64) // 10**6, 2),
        "open": np.random.randn(100) + 100,
        "high": np.random.randn(100) + 101,
        "low": np.random.randn(100) + 99,
        "close": np.random.randn(100) + 100,
        "volume": np.abs(np.random.randn(100)) * 100,
        "vwap": np.random.randn(100) + 100,
    })


class TestBarCache:
    """Tests for BarCache."""

    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initializes correctly."""
        cache = BarCache(cache_dir=temp_cache_dir)
        assert cache.cache_dir == temp_cache_dir
        assert temp_cache_dir.exists()

    def test_cache_miss_returns_none(self, temp_cache_dir):
        """Test that cache miss returns None."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        
        assert result is None

    def test_cache_put_and_get(self, temp_cache_dir, sample_bar_df):
        """Test putting and getting from cache."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        # Put data
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT", "ETHUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        
        # Get data
        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT", "ETHUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        
        assert result is not None
        assert len(result) == len(sample_bar_df)
        pd.testing.assert_frame_equal(result, sample_bar_df)

    def test_cache_key_different_symbols(self, temp_cache_dir, sample_bar_df):
        """Test that different symbols produce different cache keys."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        
        # Different symbols should miss
        result = cache.get(
            exchange="binance",
            symbols=["ETHUSDT"],  # Different symbol
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        
        assert result is None

    def test_cache_key_different_interval(self, temp_cache_dir, sample_bar_df):
        """Test that different intervals produce different cache keys."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        
        # Different interval should miss
        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=300_000,  # Different interval
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        
        assert result is None

    def test_cache_daily_files(self, temp_cache_dir, sample_bar_df):
        """Test that cache creates daily files."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 15),
        )
        
        # Check file exists with date in name
        cache_files = list(temp_cache_dir.rglob("*.parquet"))
        assert len(cache_files) == 1
        assert "2024-01-15" in cache_files[0].name

    def test_get_date_range(self, temp_cache_dir, sample_bar_df):
        """Test getting data for a date range."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        # Put data for 3 days
        for day in range(1, 4):
            cache.put(
                df=sample_bar_df,
                exchange="binance",
                symbols=["BTCUSDT"],
                interval_ms=60_000,
                data_type="aggTrades",
                market_type="futures",
                date=datetime(2024, 1, day),
            )
        
        # Get range
        result = cache.get_range(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
        )
        
        assert result is not None
        assert len(result) == len(sample_bar_df) * 3

    def test_get_date_range_partial_miss(self, temp_cache_dir, sample_bar_df):
        """Test that partial cache miss returns None for range."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        # Put data for days 1 and 3, skip day 2
        for day in [1, 3]:
            cache.put(
                df=sample_bar_df,
                exchange="binance",
                symbols=["BTCUSDT"],
                interval_ms=60_000,
                data_type="aggTrades",
                market_type="futures",
                date=datetime(2024, 1, day),
            )
        
        # Get range should return None due to missing day 2
        result = cache.get_range(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 4),
        )
        
        assert result is None

    def test_clear_cache(self, temp_cache_dir, sample_bar_df):
        """Test clearing the cache."""
        cache = BarCache(cache_dir=temp_cache_dir)
        
        cache.put(
            df=sample_bar_df,
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        
        # Clear cache
        cleared = cache.clear()
        assert cleared >= 1
        
        # Verify cache is empty
        result = cache.get(
            exchange="binance",
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            data_type="aggTrades",
            market_type="futures",
            date=datetime(2024, 1, 1),
        )
        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_cache.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create cache.py implementation**

```python
"""Daily cache layer for pre-aggregated bar data."""

import hashlib
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd


class BarCache:
    """Daily cache for pre-aggregated bar data.
    
    Cache structure:
    cache_dir/
    └── {cache_key}/
        ├── 2024-01-01.parquet
        ├── 2024-01-02.parquet
        └── ...
    
    Cache key is a hash of (exchange, symbols, interval_ms, data_type, market_type).
    Each day is stored as a separate Parquet file for efficient partial updates.
    """
    
    def __init__(self, cache_dir: Path = Path("./Data/.cache")):
        """Initialize cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _build_cache_key(
        self,
        exchange: str,
        symbols: List[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
    ) -> str:
        """Build deterministic cache key from parameters."""
        key_data = {
            "exchange": exchange,
            "symbols": sorted(symbols),
            "interval_ms": interval_ms,
            "data_type": data_type,
            "market_type": market_type,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def _get_cache_path(
        self,
        exchange: str,
        symbols: List[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        date: datetime,
    ) -> Path:
        """Get cache file path for a specific date."""
        cache_key = self._build_cache_key(
            exchange, symbols, interval_ms, data_type, market_type
        )
        date_str = date.strftime("%Y-%m-%d")
        return self.cache_dir / cache_key / f"{date_str}.parquet"
    
    def get(
        self,
        exchange: str,
        symbols: List[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Get cached data for a single date.
        
        Args:
            exchange: Exchange name
            symbols: List of symbols
            interval_ms: Bar interval in milliseconds
            data_type: Data type (e.g., 'aggTrades')
            market_type: Market type (e.g., 'futures')
            date: Date to retrieve
            
        Returns:
            DataFrame if cached, None otherwise
        """
        cache_path = self._get_cache_path(
            exchange, symbols, interval_ms, data_type, market_type, date
        )
        
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None
    
    def get_range(
        self,
        exchange: str,
        symbols: List[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Get cached data for a date range.
        
        Returns None if any date in the range is missing from cache.
        
        Args:
            exchange: Exchange name
            symbols: List of symbols
            interval_ms: Bar interval in milliseconds
            data_type: Data type
            market_type: Market type
            start_date: Start date (inclusive)
            end_date: End date (exclusive)
            
        Returns:
            Concatenated DataFrame if all dates cached, None otherwise
        """
        dfs = []
        current = start_date
        
        while current < end_date:
            df = self.get(
                exchange, symbols, interval_ms, data_type, market_type, current
            )
            if df is None:
                return None  # Cache miss
            dfs.append(df)
            current += timedelta(days=1)
        
        if not dfs:
            return None
        
        return pd.concat(dfs, ignore_index=True)
    
    def put(
        self,
        df: pd.DataFrame,
        exchange: str,
        symbols: List[str],
        interval_ms: int,
        data_type: str,
        market_type: str,
        date: datetime,
    ) -> Path:
        """Store data in cache for a single date.
        
        Args:
            df: DataFrame to cache
            exchange: Exchange name
            symbols: List of symbols
            interval_ms: Bar interval in milliseconds
            data_type: Data type
            market_type: Market type
            date: Date for this data
            
        Returns:
            Path to cached file
        """
        cache_path = self._get_cache_path(
            exchange, symbols, interval_ms, data_type, market_type, date
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return cache_path
    
    def clear(self) -> int:
        """Clear all cached data.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for cache_dir in self.cache_dir.iterdir():
            if cache_dir.is_dir():
                for f in cache_dir.glob("*.parquet"):
                    f.unlink()
                    count += 1
                cache_dir.rmdir()
        return count
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_cache.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/factorium/data/cache.py tests/data/test_cache.py
git commit -m "feat(data): add BarCache for daily pre-aggregated data caching"
```

---

### Task 6: Create New DataLoader

**Files:**
- Create: `src/factorium/data/loader_v2.py`
- Test: `tests/data/test_loader_v2.py`

**Step 1: Write the failing test**

Create `tests/data/test_loader_v2.py`:
```python
"""Tests for DataLoader (v2)."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pyarrow as pa
import pyarrow.parquet as pq

from factorium.data.loader_v2 import DataLoader
from factorium import AggBar


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with sample Parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create Hive-partitioned structure
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            for day in range(1, 8):
                partition_path = (
                    tmpdir
                    / "market=futures_um"
                    / "data_type=aggTrades"
                    / f"symbol={symbol}"
                    / "year=2024"
                    / "month=01"
                    / f"day={day:02d}"
                )
                partition_path.mkdir(parents=True, exist_ok=True)
                
                np.random.seed(42 + day + hash(symbol) % 100)
                n_trades = 500
                base_ts = int(datetime(2024, 1, day).timestamp() * 1000)
                
                df = pd.DataFrame({
                    "transact_time": base_ts + np.arange(n_trades) * 60000,
                    "price": 100.0 + np.cumsum(np.random.randn(n_trades) * 0.1),
                    "quantity": np.abs(np.random.randn(n_trades)) * 10 + 1,
                    "is_buyer_maker": np.random.choice([True, False], n_trades),
                })
                
                table = pa.Table.from_pandas(df)
                pq.write_table(table, partition_path / "data.parquet")
        
        yield tmpdir


class TestDataLoader:
    """Tests for DataLoader."""

    def test_initialization(self, temp_data_dir):
        """Test DataLoader initializes correctly."""
        loader = DataLoader(base_path=str(temp_data_dir))
        assert loader.base_path == temp_data_dir

    def test_load_aggbar_single_symbol(self, temp_data_dir):
        """Test loading AggBar for a single symbol."""
        loader = DataLoader(base_path=str(temp_data_dir), cache_enabled=False)
        
        agg = loader.load_aggbar(
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_date="2024-01-01",
            days=3,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
        )
        
        assert isinstance(agg, AggBar)
        assert "BTCUSDT" in agg.symbols
        assert len(agg) > 0

    def test_load_aggbar_multiple_symbols(self, temp_data_dir):
        """Test loading AggBar for multiple symbols."""
        loader = DataLoader(base_path=str(temp_data_dir), cache_enabled=False)
        
        agg = loader.load_aggbar(
            symbols=["BTCUSDT", "ETHUSDT"],
            interval_ms=60_000,
            start_date="2024-01-01",
            days=3,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
        )
        
        assert isinstance(agg, AggBar)
        assert set(agg.symbols) == {"BTCUSDT", "ETHUSDT"}

    def test_load_aggbar_string_symbol(self, temp_data_dir):
        """Test that string symbol is converted to list."""
        loader = DataLoader(base_path=str(temp_data_dir), cache_enabled=False)
        
        agg = loader.load_aggbar(
            symbols="BTCUSDT",  # String, not list
            interval_ms=60_000,
            start_date="2024-01-01",
            days=1,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
        )
        
        assert isinstance(agg, AggBar)
        assert "BTCUSDT" in agg.symbols

    def test_load_aggbar_different_intervals(self, temp_data_dir):
        """Test that different intervals produce different bar counts."""
        loader = DataLoader(base_path=str(temp_data_dir), cache_enabled=False)
        
        agg_1m = loader.load_aggbar(
            symbols=["BTCUSDT"],
            interval_ms=60_000,  # 1 minute
            start_date="2024-01-01",
            days=1,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
        )
        
        agg_5m = loader.load_aggbar(
            symbols=["BTCUSDT"],
            interval_ms=300_000,  # 5 minutes
            start_date="2024-01-01",
            days=1,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
        )
        
        assert len(agg_1m) > len(agg_5m)

    def test_load_aggbar_with_cache(self, temp_data_dir):
        """Test that cache is used on second load."""
        loader = DataLoader(base_path=str(temp_data_dir), cache_enabled=True)
        
        # First load (cache miss)
        agg1 = loader.load_aggbar(
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_date="2024-01-01",
            days=1,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
        )
        
        # Second load (cache hit)
        with patch.object(loader.aggregator, 'aggregate_time_bars') as mock_agg:
            agg2 = loader.load_aggbar(
                symbols=["BTCUSDT"],
                interval_ms=60_000,
                start_date="2024-01-01",
                days=1,
                exchange="binance",
                data_type="aggTrades",
                market_type="futures",
                futures_type="um",
            )
            
            # Aggregator should not be called on cache hit
            mock_agg.assert_not_called()
        
        assert len(agg1) == len(agg2)

    def test_load_aggbar_returns_correct_columns(self, temp_data_dir):
        """Test that AggBar has all required columns."""
        loader = DataLoader(base_path=str(temp_data_dir), cache_enabled=False)
        
        agg = loader.load_aggbar(
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_date="2024-01-01",
            days=1,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
        )
        
        required_cols = ["symbol", "start_time", "end_time", "open", "high", "low", "close", "volume", "vwap"]
        for col in required_cols:
            assert col in agg.cols, f"Missing column: {col}"

    def test_load_aggbar_include_buyer_seller(self, temp_data_dir):
        """Test including buyer/seller statistics."""
        loader = DataLoader(base_path=str(temp_data_dir), cache_enabled=False)
        
        agg = loader.load_aggbar(
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_date="2024-01-01",
            days=1,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            include_buyer_seller=True,
        )
        
        buyer_seller_cols = ["num_buyer", "num_seller", "num_buyer_volume", "num_seller_volume"]
        for col in buyer_seller_cols:
            assert col in agg.cols, f"Missing column: {col}"

    def test_load_aggbar_exclude_buyer_seller(self, temp_data_dir):
        """Test excluding buyer/seller statistics."""
        loader = DataLoader(base_path=str(temp_data_dir), cache_enabled=False)
        
        agg = loader.load_aggbar(
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_date="2024-01-01",
            days=1,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
            include_buyer_seller=False,
        )
        
        assert "num_buyer" not in agg.cols

    def test_unsupported_exchange_raises(self, temp_data_dir):
        """Test that unsupported exchange raises error."""
        loader = DataLoader(base_path=str(temp_data_dir))
        
        with pytest.raises(ValueError, match="Unsupported exchange"):
            loader.load_aggbar(
                symbols=["BTCUSDT"],
                interval_ms=60_000,
                start_date="2024-01-01",
                days=1,
                exchange="unknown_exchange",
            )

    def test_factor_extraction_works(self, temp_data_dir):
        """Test that factors can be extracted from loaded AggBar."""
        loader = DataLoader(base_path=str(temp_data_dir), cache_enabled=False)
        
        agg = loader.load_aggbar(
            symbols=["BTCUSDT"],
            interval_ms=60_000,
            start_date="2024-01-01",
            days=1,
            exchange="binance",
            data_type="aggTrades",
            market_type="futures",
            futures_type="um",
        )
        
        # Extract factor
        close = agg["close"]
        assert close.name == "close"
        
        # Perform operation
        momentum = close.ts_delta(5)
        assert len(momentum) == len(close)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_loader_v2.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create loader_v2.py implementation**

```python
"""High-performance data loader with DuckDB aggregation and caching."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from ..aggbar import AggBar
from .adapters.base import BaseExchangeAdapter
from .adapters.binance import BinanceAdapter
from .aggregator import BarAggregator
from .cache import BarCache


class DataLoader:
    """Unified data loader with high-performance aggregation and caching.
    
    Features:
    - DuckDB SQL aggregation (no tick data in Python memory)
    - Daily Parquet cache for fast repeated access
    - Multi-exchange support via adapters
    - Dynamic time interval adjustment
    
    Example:
        >>> loader = DataLoader()
        >>> agg = loader.load_aggbar(
        ...     symbols=["BTCUSDT", "ETHUSDT"],
        ...     interval_ms=60_000,
        ...     start_date="2024-01-01",
        ...     days=30,
        ... )
        >>> close = agg["close"]
        >>> momentum = close.ts_delta(20)
    """
    
    # Exchange adapter registry
    ADAPTERS: Dict[str, type] = {
        "binance": BinanceAdapter,
    }
    
    def __init__(
        self,
        base_path: str = "./Data",
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize DataLoader.
        
        Args:
            base_path: Base directory for data storage
            cache_enabled: Enable/disable caching
            cache_dir: Custom cache directory (default: base_path/.cache)
        """
        self.base_path = Path(base_path)
        self.aggregator = BarAggregator()
        self._adapters: Dict[str, BaseExchangeAdapter] = {}
        
        if cache_enabled:
            cache_path = Path(cache_dir) if cache_dir else self.base_path / ".cache"
            self.cache = BarCache(cache_dir=cache_path)
        else:
            self.cache = None
    
    def _get_adapter(self, exchange: str) -> BaseExchangeAdapter:
        """Get or create exchange adapter."""
        if exchange not in self._adapters:
            if exchange not in self.ADAPTERS:
                available = list(self.ADAPTERS.keys())
                raise ValueError(
                    f"Unsupported exchange: {exchange}. Available: {available}"
                )
            self._adapters[exchange] = self.ADAPTERS[exchange]()
        return self._adapters[exchange]
    
    def _resolve_date_range(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        days: Optional[int],
    ) -> tuple[datetime, datetime]:
        """Resolve date range from parameters."""
        if start_date and end_date:
            return datetime.fromisoformat(start_date), datetime.fromisoformat(end_date)
        elif start_date and days:
            start = datetime.fromisoformat(start_date)
            return start, start + timedelta(days=days)
        elif days:
            end = datetime.now()
            return end - timedelta(days=days - 1), end
        else:
            # Default: 7 days ending today
            end = datetime.now()
            return end - timedelta(days=6), end
    
    def load_aggbar(
        self,
        symbols: Union[str, List[str]],
        interval_ms: int = 60_000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        exchange: str = "binance",
        data_type: str = "aggTrades",
        market_type: str = "futures",
        futures_type: str = "um",
        use_cache: bool = True,
        include_buyer_seller: bool = True,
    ) -> AggBar:
        """Load multi-symbol OHLCV data as AggBar.
        
        Args:
            symbols: Single symbol or list of symbols
            interval_ms: Bar interval in milliseconds (dynamic)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days: Number of days (alternative to end_date)
            exchange: Exchange name
            data_type: Data type (aggTrades/trades)
            market_type: Market type (futures/spot)
            futures_type: Futures type (um/cm)
            use_cache: Use caching
            include_buyer_seller: Include buyer/seller statistics
            
        Returns:
            AggBar with aggregated OHLCV data
        """
        # Normalize symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Get adapter
        adapter = self._get_adapter(exchange)
        
        # Resolve date range
        start_dt, end_dt = self._resolve_date_range(start_date, end_date, days)
        
        # Try cache first (if enabled)
        if use_cache and self.cache:
            cached_dfs = []
            all_cached = True
            
            current = start_dt
            while current < end_dt:
                cached_df = self.cache.get(
                    exchange=exchange,
                    symbols=symbols,
                    interval_ms=interval_ms,
                    data_type=data_type,
                    market_type=market_type,
                    date=current,
                )
                if cached_df is not None:
                    cached_dfs.append(cached_df)
                else:
                    all_cached = False
                    break
                current += timedelta(days=1)
            
            if all_cached and cached_dfs:
                df = pd.concat(cached_dfs, ignore_index=True)
                return AggBar(df)
        
        # Build parquet pattern
        parquet_pattern = adapter.build_parquet_glob(
            base_path=self.base_path,
            symbols=symbols,
            data_type=data_type,
            market_type=market_type,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            futures_type=futures_type,
        )
        
        # Get column mapping
        column_mapping = adapter.get_column_mapping(data_type)
        
        # Aggregate using DuckDB
        df = self.aggregator.aggregate_time_bars(
            parquet_pattern=parquet_pattern,
            symbols=symbols,
            interval_ms=interval_ms,
            start_ts=int(start_dt.timestamp() * 1000),
            end_ts=int(end_dt.timestamp() * 1000),
            column_mapping=column_mapping,
            include_buyer_seller=include_buyer_seller,
        )
        
        # Store in cache (if enabled)
        if use_cache and self.cache:
            # Cache each day separately
            if "start_time" in df.columns:
                df_with_date = df.copy()
                df_with_date["_date"] = pd.to_datetime(
                    df_with_date["start_time"], unit="ms"
                ).dt.date
                
                for date, group in df_with_date.groupby("_date"):
                    self.cache.put(
                        df=group.drop(columns=["_date"]),
                        exchange=exchange,
                        symbols=symbols,
                        interval_ms=interval_ms,
                        data_type=data_type,
                        market_type=market_type,
                        date=datetime.combine(date, datetime.min.time()),
                    )
        
        return AggBar(df)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_loader_v2.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/factorium/data/loader_v2.py tests/data/test_loader_v2.py
git commit -m "feat(data): add DataLoader v2 with DuckDB aggregation and caching"
```

---

### Task 7: Update data/__init__.py and Integrate

**Files:**
- Modify: `src/factorium/data/__init__.py`
- Modify: `src/factorium/__init__.py`

**Step 1: Update data/__init__.py**

```python
"""Data loading and processing module.

Provides high-performance data loading with:
- DuckDB SQL aggregation (memory efficient)
- Daily Parquet caching
- Multi-exchange support
"""

from .loader_v2 import DataLoader
from .aggregator import BarAggregator
from .cache import BarCache
from .adapters import BaseExchangeAdapter, ColumnMapping, BinanceAdapter
from .parquet import csv_to_parquet, read_hive_parquet, build_hive_path, get_market_string, BINANCE_COLUMNS

# Legacy import (deprecated)
from .loader import BinanceDataLoader
from .downloader import BinanceDataDownloader

__all__ = [
    # New API
    "DataLoader",
    "BarAggregator",
    "BarCache",
    "BaseExchangeAdapter",
    "ColumnMapping",
    "BinanceAdapter",
    # Utilities
    "csv_to_parquet",
    "read_hive_parquet",
    "build_hive_path",
    "get_market_string",
    "BINANCE_COLUMNS",
    # Legacy (deprecated)
    "BinanceDataLoader",
    "BinanceDataDownloader",
]
```

**Step 2: Update src/factorium/__init__.py to export DataLoader**

Read the current file first, then add DataLoader to exports.

**Step 3: Run all data tests**

Run: `uv run pytest tests/data/ tests/test_data_loader.py -v`
Expected: PASS (all tests)

**Step 4: Commit**

```bash
git add src/factorium/data/__init__.py src/factorium/__init__.py
git commit -m "feat(data): integrate DataLoader v2 into public API"
```

---

### Task 8: Run Full Test Suite and Verify

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 2: Run linting**

Run: `uv run ruff check src/factorium/data/`
Expected: No errors (or fix any that appear)

**Step 3: Run type checking**

Run: `uv run mypy src/factorium/data/`
Expected: No errors (or acceptable with ignore_missing_imports)

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address linting and type errors in data module"
```

---

## Phase 2: Factor Computation Optimization

### Task 9: Create Polars Engine

**Files:**
- Create: `src/factorium/factors/engine.py`
- Test: `tests/factors/test_engine.py`

**Step 1: Create test directory**

Run: `mkdir -p tests/factors`

**Step 2: Write the failing test**

Create `tests/factors/__init__.py`:
```python
# Tests for factors module
```

Create `tests/factors/test_engine.py`:
```python
"""Tests for PolarsEngine."""

import pytest
import polars as pl
import pandas as pd
import numpy as np

from factorium.factors.engine import PolarsEngine


@pytest.fixture
def sample_factor_df():
    """Create sample factor data as Polars DataFrame."""
    np.random.seed(42)
    n_rows = 100
    
    dates = pd.date_range("2024-01-01", periods=50, freq="1min")
    timestamps = dates.astype(np.int64) // 10**6
    
    return pl.DataFrame({
        "start_time": pl.Series(list(timestamps) * 2),
        "end_time": pl.Series(list(timestamps + 60000) * 2),
        "symbol": ["BTCUSDT"] * 50 + ["ETHUSDT"] * 50,
        "factor": pl.Series(np.random.randn(n_rows)),
    })


class TestPolarsEngine:
    """Tests for PolarsEngine."""

    def test_ts_mean(self, sample_factor_df):
        """Test rolling mean."""
        result = PolarsEngine.ts_mean(sample_factor_df, window=5)
        
        assert isinstance(result, pl.DataFrame)
        assert "factor" in result.columns
        assert len(result) == len(sample_factor_df)
        
        # First 4 values per symbol should be null (window not full)
        btc_data = result.filter(pl.col("symbol") == "BTCUSDT")
        assert btc_data["factor"][:4].null_count() == 4

    def test_ts_std(self, sample_factor_df):
        """Test rolling standard deviation."""
        result = PolarsEngine.ts_std(sample_factor_df, window=5)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_factor_df)

    def test_ts_shift(self, sample_factor_df):
        """Test shift operation."""
        result = PolarsEngine.ts_shift(sample_factor_df, period=3)
        
        assert isinstance(result, pl.DataFrame)
        
        # First 3 values per symbol should be null
        btc_data = result.filter(pl.col("symbol") == "BTCUSDT")
        assert btc_data["factor"][:3].null_count() == 3

    def test_ts_diff(self, sample_factor_df):
        """Test diff operation."""
        result = PolarsEngine.ts_diff(sample_factor_df, period=1)
        
        assert isinstance(result, pl.DataFrame)
        
        # First value per symbol should be null
        btc_data = result.filter(pl.col("symbol") == "BTCUSDT")
        assert btc_data["factor"][0] is None

    def test_cs_rank(self, sample_factor_df):
        """Test cross-sectional rank."""
        result = PolarsEngine.cs_rank(sample_factor_df)
        
        assert isinstance(result, pl.DataFrame)
        
        # Ranks should be between 0 and 1
        assert result["factor"].min() >= 0
        assert result["factor"].max() <= 1

    def test_cs_zscore(self, sample_factor_df):
        """Test cross-sectional z-score."""
        result = PolarsEngine.cs_zscore(sample_factor_df)
        
        assert isinstance(result, pl.DataFrame)
        
        # Z-scores for each time should have mean ~0
        for end_time in result["end_time"].unique():
            time_data = result.filter(pl.col("end_time") == end_time)
            mean = time_data["factor"].mean()
            assert abs(mean) < 0.1  # Close to 0

    def test_to_pandas(self, sample_factor_df):
        """Test conversion to Pandas."""
        result = PolarsEngine.to_pandas(sample_factor_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_factor_df)

    def test_from_pandas(self):
        """Test conversion from Pandas."""
        pdf = pd.DataFrame({
            "start_time": [1, 2, 3],
            "end_time": [2, 3, 4],
            "symbol": ["A", "A", "A"],
            "factor": [1.0, 2.0, 3.0],
        })
        
        result = PolarsEngine.from_pandas(pdf)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(pdf)
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/factors/test_engine.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 4: Create engine.py implementation**

```python
"""Polars-based computation engine for high-performance factor operations."""

import polars as pl
import pandas as pd


class PolarsEngine:
    """High-performance factor computation engine using Polars.
    
    Provides efficient implementations of:
    - Time-series operations (rolling mean, std, shift, diff)
    - Cross-sectional operations (rank, zscore)
    - Data format conversion (Polars <-> Pandas)
    
    All operations are designed to work with the standard factor DataFrame format:
    - start_time: int64 (milliseconds)
    - end_time: int64 (milliseconds)
    - symbol: str
    - factor: float64
    """
    
    @staticmethod
    def ts_mean(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling mean with strict window (NaN if window not full).
        
        Args:
            df: Input DataFrame
            window: Rolling window size
            symbol_col: Column name for symbol grouping
            value_col: Column name for values
            
        Returns:
            DataFrame with rolling mean applied
        """
        return df.with_columns(
            pl.when(
                pl.col(value_col).rolling_count(window_size=window).over(symbol_col) >= window
            )
            .then(pl.col(value_col).rolling_mean(window_size=window).over(symbol_col))
            .otherwise(None)
            .alias(value_col)
        )
    
    @staticmethod
    def ts_std(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling standard deviation with strict window."""
        return df.with_columns(
            pl.when(
                pl.col(value_col).rolling_count(window_size=window).over(symbol_col) >= window
            )
            .then(pl.col(value_col).rolling_std(window_size=window).over(symbol_col))
            .otherwise(None)
            .alias(value_col)
        )
    
    @staticmethod
    def ts_sum(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling sum with strict window."""
        return df.with_columns(
            pl.when(
                pl.col(value_col).rolling_count(window_size=window).over(symbol_col) >= window
            )
            .then(pl.col(value_col).rolling_sum(window_size=window).over(symbol_col))
            .otherwise(None)
            .alias(value_col)
        )
    
    @staticmethod
    def ts_min(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling minimum with strict window."""
        return df.with_columns(
            pl.when(
                pl.col(value_col).rolling_count(window_size=window).over(symbol_col) >= window
            )
            .then(pl.col(value_col).rolling_min(window_size=window).over(symbol_col))
            .otherwise(None)
            .alias(value_col)
        )
    
    @staticmethod
    def ts_max(
        df: pl.DataFrame,
        window: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Rolling maximum with strict window."""
        return df.with_columns(
            pl.when(
                pl.col(value_col).rolling_count(window_size=window).over(symbol_col) >= window
            )
            .then(pl.col(value_col).rolling_max(window_size=window).over(symbol_col))
            .otherwise(None)
            .alias(value_col)
        )
    
    @staticmethod
    def ts_shift(
        df: pl.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Shift values by period within each symbol."""
        return df.with_columns(
            pl.col(value_col).shift(period).over(symbol_col).alias(value_col)
        )
    
    @staticmethod
    def ts_diff(
        df: pl.DataFrame,
        period: int,
        symbol_col: str = "symbol",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Difference from period ago within each symbol."""
        return df.with_columns(
            pl.col(value_col).diff(period).over(symbol_col).alias(value_col)
        )
    
    @staticmethod
    def cs_rank(
        df: pl.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional percentile rank (0 to 1)."""
        return df.with_columns(
            (pl.col(value_col).rank(method="min").over(time_col) - 1)
            .truediv(pl.col(value_col).count().over(time_col) - 1)
            .alias(value_col)
        )
    
    @staticmethod
    def cs_zscore(
        df: pl.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional z-score."""
        return df.with_columns(
            ((pl.col(value_col) - pl.col(value_col).mean().over(time_col))
             / pl.col(value_col).std().over(time_col))
            .alias(value_col)
        )
    
    @staticmethod
    def cs_demean(
        df: pl.DataFrame,
        time_col: str = "end_time",
        value_col: str = "factor",
    ) -> pl.DataFrame:
        """Cross-sectional demean (subtract mean)."""
        return df.with_columns(
            (pl.col(value_col) - pl.col(value_col).mean().over(time_col))
            .alias(value_col)
        )
    
    @staticmethod
    def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
        """Convert Polars DataFrame to Pandas."""
        return df.to_pandas()
    
    @staticmethod
    def from_pandas(df: pd.DataFrame) -> pl.DataFrame:
        """Convert Pandas DataFrame to Polars."""
        return pl.from_pandas(df)
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/factors/test_engine.py -v`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/factorium/factors/engine.py tests/factors/
git commit -m "feat(factors): add PolarsEngine for high-performance factor operations"
```

---

### Task 10: Refactor TimeSeriesOpsMixin to Use PolarsEngine

**Files:**
- Modify: `src/factorium/factors/mixins/ts_ops.py`
- Modify: `src/factorium/factors/base.py`
- Test: Run existing tests to verify backward compatibility

**Step 1: Read current implementations**

Read `src/factorium/factors/mixins/ts_ops.py` and `src/factorium/factors/base.py` to understand current structure.

**Step 2: Modify base.py to add Polars support**

Add to `BaseFactor.__init__`:
```python
self._pl_data: Optional[pl.DataFrame] = None  # Lazy Polars cache
```

Add methods:
```python
def _to_polars(self) -> pl.DataFrame:
    """Convert to Polars (lazy, cached)."""
    if self._pl_data is None:
        from ..factors.engine import PolarsEngine
        self._pl_data = PolarsEngine.from_pandas(self._data)
    return self._pl_data

def _from_polars(self, pl_df: pl.DataFrame, name: str) -> Self:
    """Create new Factor from Polars DataFrame."""
    from ..factors.engine import PolarsEngine
    return self.__class__(PolarsEngine.to_pandas(pl_df), name)
```

**Step 3: Refactor ts_ops.py methods to use PolarsEngine**

Example refactor for `ts_mean`:
```python
def ts_mean(self, window: int) -> Self:
    """Rolling mean using Polars engine."""
    self._validate_window(window)
    
    from ..engine import PolarsEngine
    pl_df = self._to_polars()
    result = PolarsEngine.ts_mean(pl_df, window)
    
    return self._from_polars(result, f"ts_mean({self.name},{window})")
```

**Step 4: Run existing tests**

Run: `uv run pytest tests/mixins/test_ts_ops.py -v`
Expected: PASS (backward compatible)

**Step 5: Commit**

```bash
git add src/factorium/factors/base.py src/factorium/factors/mixins/ts_ops.py
git commit -m "refactor(factors): use PolarsEngine for time-series operations"
```

---

### Task 11: Refactor CrossSectionalOpsMixin to Use PolarsEngine

**Files:**
- Modify: `src/factorium/factors/mixins/cs_ops.py`

**Step 1: Refactor cs_ops.py methods to use PolarsEngine**

Similar to Task 10, refactor `cs_rank`, `cs_zscore`, `cs_demean` to use PolarsEngine.

**Step 2: Run existing tests**

Run: `uv run pytest tests/mixins/test_cs_ops.py -v`
Expected: PASS (backward compatible)

**Step 3: Commit**

```bash
git add src/factorium/factors/mixins/cs_ops.py
git commit -m "refactor(factors): use PolarsEngine for cross-sectional operations"
```

---

## Phase 3: Documentation and Cleanup

### Task 12: Update Documentation

**Files:**
- Modify: `docs/user-guide/factor.md`
- Create: `docs/user-guide/data-loading.md`

**Step 1: Update factor.md with performance notes**

Add section about internal Polars engine.

**Step 2: Create data-loading.md**

Document new DataLoader API:
- Basic usage
- Caching behavior
- Multi-exchange support
- Performance tips

**Step 3: Commit**

```bash
git add docs/
git commit -m "docs: update documentation for optimized data pipeline"
```

---

### Task 13: Run Full Test Suite and Performance Benchmark

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 2: Create simple benchmark script**

Create `scripts/benchmark_loader.py`:
```python
"""Simple benchmark for data loading performance."""

import time
from factorium import DataLoader

def benchmark():
    loader = DataLoader(base_path="./Data", cache_enabled=True)
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    # First load (cache miss)
    start = time.time()
    agg = loader.load_aggbar(
        symbols=symbols,
        interval_ms=60_000,
        start_date="2024-01-01",
        days=7,
    )
    first_load = time.time() - start
    print(f"First load (cache miss): {first_load:.2f}s, {len(agg)} rows")
    
    # Second load (cache hit)
    start = time.time()
    agg = loader.load_aggbar(
        symbols=symbols,
        interval_ms=60_000,
        start_date="2024-01-01",
        days=7,
    )
    second_load = time.time() - start
    print(f"Second load (cache hit): {second_load:.2f}s, {len(agg)} rows")
    
    # Factor operation
    close = agg["close"]
    start = time.time()
    result = close.ts_mean(20).ts_std(20).cs_rank()
    factor_ops = time.time() - start
    print(f"Factor operations: {factor_ops:.2f}s")

if __name__ == "__main__":
    benchmark()
```

**Step 3: Commit**

```bash
git add scripts/
git commit -m "chore: add benchmark script for data loading"
```

---

### Task 14: Final Review and Cleanup

**Step 1: Run linting on all new files**

Run: `uv run ruff check src/factorium/`
Run: `uv run ruff format src/factorium/`

**Step 2: Run type checking**

Run: `uv run mypy src/factorium/`

**Step 3: Run full test suite**

Run: `uv run pytest -v --cov=src/factorium`

**Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "chore: final cleanup and formatting"
```

---

## Summary

This plan implements a complete data pipeline optimization with:

1. **Phase 1 (Data Loading)**: 8 tasks
   - DuckDB SQL aggregation
   - Daily Parquet caching
   - Exchange adapter abstraction
   - New DataLoader API

2. **Phase 2 (Factor Computation)**: 3 tasks
   - Polars-based computation engine
   - Refactored time-series operations
   - Refactored cross-sectional operations

3. **Phase 3 (Documentation)**: 3 tasks
   - Updated documentation
   - Benchmark script
   - Final cleanup

**Expected Performance Improvements:**
- Load time: 10-60x faster (with caching)
- Memory usage: 5-10x lower
- Factor operations: 10-20x faster
- 100 symbols: Now feasible (was OOM)
