"""
Aggregated bar data container for multi-symbol panel data.

AggBar provides a unified interface for working with OHLCV data
across multiple symbols in long format.
"""

import pandas as pd
import polars as pl
import numpy as np
from typing import Union, List, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    from .factors.core import Factor
    from .data.metadata import AggBarMetadata


class AggBar:
    """
    Multi-symbol bar data container.

    Stores OHLCV data for multiple symbols in long format with columns:
    start_time, end_time, symbol, open, high, low, close, volume, ...

    Args:
        data: Either a list of BaseBar objects, a Pandas DataFrame, or a Polars DataFrame
              with at least start_time, end_time, symbol columns
        metadata: Optional AggBarMetadata for pre-computed symbols and time range

    Example:
        >>> from factorium import AggBar
        >>> # Load bar data from DataFrame or other sources
        >>> agg = AggBar(df)
        >>> close_factor = agg['close']  # Returns Factor
    """

    def __init__(
        self,
        data: Union[List["BaseBar"], pd.DataFrame, pl.DataFrame],
        metadata: Optional["AggBarMetadata"] = None,
    ):
        # Import here to avoid circular imports
        from .data.metadata import AggBarMetadata

        # Convert to Polars
        if isinstance(data, list):
            pdf = pd.concat([bar.bars for bar in data])
            self._data = pl.from_pandas(pdf)
        elif isinstance(data, pd.DataFrame):
            self._data = pl.from_pandas(data)
        elif isinstance(data, pl.DataFrame):
            self._data = data
        else:
            raise TypeError(f"Invalid data type: {type(data)}")

        # Validate columns
        required = {"start_time", "end_time", "symbol"}
        missing = required - set(self._data.columns)
        if missing:
            raise ValueError(f"DataFrame must contain columns: {missing}")

        # Sort by end_time, symbol
        self._data = self._data.sort(["end_time", "symbol"])

        # Metadata
        if metadata is None:
            self._metadata = self._compute_metadata()
        else:
            self._metadata = metadata

    def _compute_metadata(self) -> "AggBarMetadata":
        """Compute metadata from data."""
        from .data.metadata import AggBarMetadata

        if self._data.is_empty():
            return AggBarMetadata(symbols=[], min_time=0, max_time=0, num_rows=0)

        return AggBarMetadata(
            symbols=sorted(self._data["symbol"].unique().to_list()),
            min_time=self._data["start_time"].min(),
            max_time=self._data["end_time"].max(),
            num_rows=len(self._data),
        )

    @classmethod
    def from_bars(cls, bars: List) -> "AggBar":
        """Create AggBar from a list of BaseBar objects."""
        return cls(bars)

    @classmethod
    def from_df(cls, df: Union[pd.DataFrame, pl.DataFrame]) -> "AggBar":
        """Create AggBar from a DataFrame."""
        return cls(df)

    @classmethod
    def from_csv(cls, path: Path) -> "AggBar":
        """Create AggBar from a CSV file."""
        df = pd.read_csv(path)
        return cls(df)

    def to_df(self) -> pd.DataFrame:
        """Return the data as a Pandas DataFrame copy."""
        return self._data.to_pandas()

    @property
    def data(self) -> pd.DataFrame:
        """Return data as Pandas DataFrame for backward compatibility."""
        return self.to_df()

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

            # Create factor data with required columns
            factor_df = self._data.select(["start_time", "end_time", "symbol", key])
            # Rename the value column to 'factor'
            factor_df = factor_df.rename({key: "factor"})
            return Factor(factor_df, name=key)

        elif isinstance(key, list):
            cols = ["start_time", "end_time", "symbol"] + [
                c for c in key if c not in ["start_time", "end_time", "symbol"]
            ]
            return AggBar(self._data.select(cols).clone())
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
                return int(pd.to_datetime(value).timestamp() * 1000)
            elif isinstance(value, (int, np.integer)):
                value_int = int(value)
                # Handle different timestamp formats
                # 13 digits or more = milliseconds
                # 10 digits = seconds (convert to ms)
                # Less than 10 = assume milliseconds already
                if len(str(abs(value_int))) >= 13:
                    return value_int
                elif len(str(abs(value_int))) == 10:
                    return value_int * 1000
                else:
                    # Assume it's already in milliseconds
                    return value_int
            elif isinstance(value, datetime):
                return int(value.timestamp() * 1000)
            else:
                raise ValueError(f"Cannot convert {value} to timestamp")

        start_ts = convert_timestamp(start)
        end_ts = convert_timestamp(end)

        if symbols is None:
            symbols = self.symbols

        # Build filter condition
        cond = self._data["symbol"].is_in(symbols)
        if start_ts is not None:
            cond = cond & (self._data["start_time"] >= start_ts)
        if end_ts is not None:
            cond = cond & (self._data["end_time"] <= end_ts)

        return AggBar(self._data.filter(cond))

    @property
    def cols(self) -> List[str]:
        """Return list of column names."""
        return self._data.columns

    @property
    def symbols(self) -> List[str]:
        """Return list of unique symbols from metadata."""
        return self._metadata.symbols

    @property
    def metadata(self) -> "AggBarMetadata":
        """Return the metadata."""
        return self._metadata

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Return unique timestamps from start_time and end_time."""
        ts1 = pd.to_datetime(self._data["start_time"].to_numpy(), unit="ms").unique()
        ts2 = pd.to_datetime(self._data["end_time"].to_numpy(), unit="ms").unique()
        all_ts = np.unique(np.concatenate([ts1, ts2]))
        return pd.DatetimeIndex(all_ts)

    def info(self) -> pd.DataFrame:
        """
        Get summary information for each symbol.

        Returns:
            DataFrame with num_kbar, start_time, end_time, num_nan per symbol
        """
        # Convert to Pandas to use groupby (Polars groupby is different)
        pdf = self._data.to_pandas()
        grouped = pdf.groupby("symbol")

        return pd.DataFrame(
            {
                "num_kbar": grouped.size(),
                "start_time": grouped["start_time"]
                .min()
                .apply(lambda x: pd.to_datetime(x, unit="ms") if pd.notnull(x) else pd.NaT),
                "end_time": grouped["end_time"]
                .max()
                .apply(lambda x: pd.to_datetime(x, unit="ms") if pd.notnull(x) else pd.NaT),
                "num_nan": grouped.apply(lambda df: df.isna().sum().sum(), include_groups=False).astype(int),
            }
        )

    def copy(self) -> "AggBar":
        """Return a copy of this AggBar."""
        return AggBar(self._data.clone())

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self):
        n_symbols = len(self.symbols)
        time_range = f"{self.timestamps.min().strftime('%Y-%m-%d %H:%M:%S')} - {self.timestamps.max().strftime('%Y-%m-%d %H:%M:%S')}"

        return f"AggBar: {len(self)} rows, {len(self.cols)} columns, symbols={n_symbols}, time_range={time_range}"
