"""
Aggregated bar data container for multi-symbol panel data.

AggBar provides a unified interface for working with OHLCV data
across multiple symbols in long format.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    from .bar import BaseBar
    from .factors.core import Factor


class AggBar:
    """
    Multi-symbol bar data container.
    
    Stores OHLCV data for multiple symbols in long format with columns:
    start_time, end_time, symbol, open, high, low, close, volume, ...
    
    Args:
        bars: Either a list of BaseBar objects or a long-format DataFrame
              with at least start_time, end_time, symbol columns
    
    Example:
        >>> from factorium import AggBar, TimeBar
        >>> bars = [TimeBar(df1), TimeBar(df2)]
        >>> agg = AggBar(bars)
        >>> close_factor = agg['close']  # Returns Factor
    """
    
    def __init__(self, bars: Union[List["BaseBar"], pd.DataFrame]):
        if isinstance(bars, list):
            self.data = pd.concat([bar.bars for bar in bars])
        elif isinstance(bars, pd.DataFrame):
            if "start_time" not in bars.columns or "end_time" not in bars.columns or "symbol" not in bars.columns:
                raise ValueError("DataFrame must contain columns: start_time, end_time, symbol")
            
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index()
            self.data = bars
        else:
            raise TypeError(f"Invalid bars type: {type(bars)}")
        
        self.data = self.data.sort_values(by=['end_time', 'symbol']).reset_index(drop=True)
        
    @classmethod
    def from_bars(cls, bars: List["BaseBar"]) -> 'AggBar':
        """Create AggBar from a list of BaseBar objects."""
        return cls(bars)
    
    @classmethod
    def from_df(cls, df: pd.DataFrame) -> 'AggBar':
        """Create AggBar from a DataFrame."""
        return cls(df)
    
    @classmethod
    def from_csv(cls, path: Path) -> 'AggBar':
        """Create AggBar from a CSV file."""
        df = pd.read_csv(path)
        return cls(df)
    
    def to_df(self) -> pd.DataFrame:
        """Return the data as a DataFrame copy."""
        return self.data.copy()
    
    def to_csv(self, path: Path) -> Path:
        """Save data to a CSV file."""
        self.data.to_csv(path, index=False)
        return path
    
    def to_parquet(self, path: Path) -> Path:
        """Save data to a Parquet file."""
        self.data.to_parquet(path, index=False)
        return path
    
    def __getitem__(self, key: Union[str, List[str]]) -> Union["Factor", 'AggBar']:
        """
        Get a column as a Factor or multiple columns as a new AggBar.
        
        Args:
            key: Column name (str) or list of column names
            
        Returns:
            Factor if single column, AggBar if multiple columns
        """
        if isinstance(key, str):
            if key not in self.data.columns:
                raise KeyError(f"Column {key} not found in the dataframe")
            
            # Lazy import to avoid circular dependency
            from .factors.core import Factor
            
            factor_df = self.data[["start_time", "end_time", "symbol", key]].copy()
            factor_df.columns = ["start_time", "end_time", "symbol", "factor"]
            return Factor(factor_df, name=key)
        
        elif isinstance(key, list):
            cols = ["start_time", "end_time", "symbol"] + [c for c in key if c not in ["start_time", "end_time", "symbol"]]
            return AggBar(self.data[cols].copy())
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
        
    def slice(
        self,
        start: Optional[Union[datetime, int, str]] = None,
        end: Optional[Union[datetime, int, str]] = None,
        symbols: Optional[List[str]] = None
    ) -> 'AggBar':
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
                if len(str(value_int)) == 13:
                    return value_int
                elif len(str(value_int)) == 10:
                    return value_int * 1000
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

        cond = self.data["symbol"].isin(symbols)
        if start_ts is not None:
            cond = cond & (self.data["start_time"] >= start_ts)
        if end_ts is not None:
            cond = cond & (self.data["end_time"] <= end_ts)
        
        return AggBar(self.data[cond])
    
    @property
    def cols(self) -> List[str]:
        """Return list of column names."""
        return self.data.columns.tolist()
    
    @property
    def symbols(self) -> List[str]:
        """Return list of unique symbols."""
        return self.data["symbol"].unique().tolist()
    
    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Return unique timestamps from start_time and end_time."""
        ts1 = pd.to_datetime(self.data["start_time"], unit='ms').unique()
        ts2 = pd.to_datetime(self.data["end_time"], unit='ms').unique()
        all_ts = np.unique(np.concatenate([ts1, ts2]))
        return pd.DatetimeIndex(all_ts)
    
    def info(self) -> pd.DataFrame:
        """
        Get summary information for each symbol.
        
        Returns:
            DataFrame with num_kbar, start_time, end_time, num_nan per symbol
        """
        grouped = self.data.groupby("symbol")
        
        return pd.DataFrame({
            'num_kbar': grouped.size(),
            'start_time': grouped['start_time'].min().apply(lambda x: pd.to_datetime(x, unit='ms') if pd.notnull(x) else pd.NaT),
            'end_time': grouped['end_time'].max().apply(lambda x: pd.to_datetime(x, unit='ms') if pd.notnull(x) else pd.NaT),
            'num_nan': grouped.apply(lambda df: df.isna().sum().sum(), include_groups=False).astype(int)
        })
        
    def copy(self) -> 'AggBar':
        """Return a copy of this AggBar."""
        return AggBar(self.data.copy())
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self):
        n_symbols = len(self.symbols)
        time_range = f"{self.timestamps.min().strftime('%Y-%m-%d %H:%M:%S')} - {self.timestamps.max().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return f"AggBar: {len(self)} rows, {len(self.cols)} columns, symbols={n_symbols}, time_range={time_range}"
