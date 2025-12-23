"""
Bar data structures for different sampling methods.

Provides various bar types for financial data:
- TimeBar: Fixed time interval bars
- TickBar: Fixed number of ticks per bar
- VolumeBar: Fixed volume per bar
- DollarBar: Fixed dollar volume per bar
"""

import pandas as pd
import numpy as np
import numba
from abc import ABC, abstractmethod
from typing import Dict, Callable


class BaseBar(ABC):
    """
    Abstract base class for bar data.
    
    Args:
        df: DataFrame with tick/trade data
        timestamp_col: Column name for timestamps
        price_col: Column name for prices
        volume_col: Column name for volumes
        interval: Bar interval (meaning depends on bar type)
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        timestamp_col: str = 'ts_init',
        price_col: str = 'price',
        volume_col: str = 'size',
        interval: int = 100000
    ):
        self.interval = interval
        self._df = df
        self.timestamp_col = timestamp_col
        self.price_col = price_col
        self.volume_col = volume_col
        
        self._bar = self._create_bars()
        
        # Ensure symbol column (if exists) is the first column in bars
        if 'symbol' in self._df.columns and not self._bar.empty:
            symbol_value = self._df['symbol'].iloc[0]
            self._bar.insert(0, 'symbol', symbol_value)
    
    def _create_bars(self) -> pd.DataFrame:
        """Aggregate the K bar data according to the group_id"""
        group_id = self._get_group_idx()
        return self._aggregate_by_group(group_id)
    
    def _aggregate_by_group(self, group_id: np.ndarray) -> pd.DataFrame:
        """Generic grouping aggregation logic"""
        result = self._df.groupby(group_id).agg(
            start_time=(self.timestamp_col, "first"),
            end_time=(self.timestamp_col, "last"),
            open=(self.price_col, "first"),
            high=(self.price_col, "max"),
            low=(self.price_col, "min"),
            close=(self.price_col, "last"),
            volume=(self.volume_col, "sum")
        )
        
        # Add buyer/seller statistics if is_buyer_maker column exists
        if 'is_buyer_maker' in self._df.columns:
            df_temp = self._df.assign(
                _is_buyer=(~self._df['is_buyer_maker']).astype(int),
                _is_seller=self._df['is_buyer_maker'].astype(int),
                _buyer_volume=self._df[self.volume_col] * (~self._df['is_buyer_maker']),
                _seller_volume=self._df[self.volume_col] * self._df['is_buyer_maker']
            )
            
            buyer_seller_agg = df_temp.groupby(group_id).agg(
                num_buyer=('_is_buyer', 'sum'),
                num_seller=('_is_seller', 'sum'),
                num_buyer_volume=('_buyer_volume', 'sum'),
                num_seller_volume=('_seller_volume', 'sum')
            )
            
            result = pd.concat([result, buyer_seller_agg], axis=1)
        
        return result
    
    @abstractmethod
    def _get_group_idx(self) -> np.ndarray:
        """Return group indices for aggregation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_group_idx")
    
    def apply(self, transformations: Dict[str, Callable]) -> 'BaseBar':
        """
        Apply feature transformations to the bars.
        
        Args:
            transformations: Dict mapping feature names to transformation functions.
                            Each function receives the bars DataFrame and returns a Series.
        
        Returns:
            self for method chaining
            
        Example:
            bar.apply({
                'forward_return_5': lambda bars: (bars['close'].shift(-5) - bars['close']) / bars['close'],
                'sma_20': lambda bars: bars['close'].rolling(20).mean(),
            })
        """
        bars = self._bar
        
        for feature_name, func in transformations.items():
            try:
                result = func(bars)
                
                if isinstance(result, pd.Series):
                    if len(result) == len(self._bar):
                        self._bar[feature_name] = result.values
                    else:
                        raise ValueError(f"Series length ({len(result)}) does not match bars length ({len(self._bar)})")
                elif np.isscalar(result):
                    self._bar[feature_name] = result
                else:
                    raise ValueError(f"Function must return pandas Series or scalar value, got {type(result)}")
                    
            except Exception as e:
                raise ValueError(f"Apply transformation '{feature_name}' failed: {e}")
        
        return self

    @property
    def bars(self) -> pd.DataFrame:
        """Return the aggregated bar data."""
        return self._bar
    
    def __len__(self) -> int:
        return len(self._bar)


class TimeBar(BaseBar):
    """
    Time-based bars with fixed time intervals.
    
    Args:
        df: DataFrame with tick/trade data
        timestamp_col: Column name for timestamps (milliseconds)
        price_col: Column name for prices
        volume_col: Column name for volumes
        interval_ms: Time interval in milliseconds (default: 60000 = 1 minute)
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        timestamp_col: str = 'ts_init',
        price_col: str = 'price',
        volume_col: str = 'size',
        interval_ms: int = 60_000
    ):
        super().__init__(df, timestamp_col, price_col, volume_col, interval_ms)
    
    def _get_group_idx(self) -> np.ndarray:
        return self._group_trades_by_time(
            self._df[self.timestamp_col].values,
            self.interval
        )
    
    def _aggregate_by_group(self, group_id: np.ndarray) -> pd.DataFrame:
        result = self._df.groupby(group_id).agg(
            open=(self.price_col, "first"),
            high=(self.price_col, "max"),
            low=(self.price_col, "min"),
            close=(self.price_col, "last"),
            volume=(self.volume_col, "sum")
        )
        
        if 'is_buyer_maker' in self._df.columns:
            df_temp = self._df.assign(
                _is_buyer=(~self._df['is_buyer_maker']).astype(int),
                _is_seller=self._df['is_buyer_maker'].astype(int),
                _buyer_volume=self._df[self.volume_col] * (~self._df['is_buyer_maker']),
                _seller_volume=self._df[self.volume_col] * self._df['is_buyer_maker']
            )
            
            buyer_seller_agg = df_temp.groupby(group_id).agg(
                num_buyer=('_is_buyer', 'sum'),
                num_seller=('_is_seller', 'sum'),
                num_buyer_volume=('_buyer_volume', 'sum'),
                num_seller_volume=('_seller_volume', 'sum')
            )
            
            result = pd.concat([result, buyer_seller_agg], axis=1)
        
        timestamps = self._df[self.timestamp_col].values
        first_timestamp = timestamps[0]
        start_time_base = (first_timestamp // self.interval) * self.interval
        
        bar_indices = result.index.values
        start_times = start_time_base + bar_indices * self.interval
        end_times = start_times + self.interval
        
        result['start_time'] = start_times
        result['end_time'] = end_times
        
        base_columns = ['start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume']
        extra_columns = [col for col in result.columns if col not in base_columns]
        result = result[base_columns + extra_columns]
        
        return result.reset_index(drop=True)
    
    @staticmethod
    @numba.jit(nopython=True)
    def _group_trades_by_time(timestamps: np.ndarray, interval_ms: int) -> np.ndarray:
        bar_ids = np.zeros(len(timestamps), dtype=np.int64)
        
        if len(timestamps) == 0:
            return bar_ids
            
        first_timestamp = timestamps[0]
        start_time = (first_timestamp // interval_ms) * interval_ms
        
        for i in range(len(timestamps)):
            bar_id = (timestamps[i] - start_time) // interval_ms
            bar_ids[i] = bar_id
            
        return bar_ids


class TickBar(BaseBar):
    """
    Tick-based bars with fixed number of ticks per bar.
    
    Args:
        df: DataFrame with tick/trade data
        timestamp_col: Column name for timestamps
        price_col: Column name for prices
        volume_col: Column name for volumes
        interval_ticks: Number of ticks per bar
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'ts_init',
        price_col: str = 'price',
        volume_col: str = 'size',
        interval_ticks: int = 100000,
    ):
        super().__init__(df, timestamp_col, price_col, volume_col, interval_ticks)
    
    def _get_group_idx(self) -> np.ndarray:
        return self._group_trades_by_tick(self._df.index.values, self.interval)
    
    @staticmethod
    @numba.jit(nopython=True) 
    def _group_trades_by_tick(df_values: np.ndarray, interval_ticks: int) -> np.ndarray:
        bar_ids = np.zeros(df_values.shape[0], dtype=np.int64)
        current_tick = 0
        bar_id = 0
        
        for i in range(len(df_values)):
            if current_tick == interval_ticks:
                bar_id += 1
                current_tick = 0
            bar_ids[i] = bar_id
            current_tick += 1
        return bar_ids


class VolumeBar(BaseBar):
    """
    Volume-based bars with fixed volume per bar.
    
    Args:
        df: DataFrame with tick/trade data
        timestamp_col: Column name for timestamps
        price_col: Column name for prices
        volume_col: Column name for volumes
        interval_volume: Target volume per bar
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'time',
        price_col: str = 'price',
        volume_col: str = 'quote_qty',
        interval_volume: int = 100000,
    ):
        super().__init__(df, timestamp_col, price_col, volume_col, interval_volume)
    
    def _create_bars(self) -> pd.DataFrame:
        group_id = self._get_group_idx()
        return self._aggregate_by_group(group_id)
    
    def _get_group_idx(self) -> np.ndarray:
        return self._group_trades_by_volume(
            self._df[self.volume_col].values, 
            self.interval
        )
    
    @staticmethod
    @numba.jit(nopython=True)
    def _group_trades_by_volume(df_values: np.ndarray, target_volume: float) -> np.ndarray:
        bar_ids = np.zeros(len(df_values), dtype=np.int64)
        current_volume = 0
        bar_id = 0
        
        for i in range(len(df_values)):
            current_volume += df_values[i]
            bar_ids[i] = bar_id
            if current_volume >= target_volume:
                current_volume = 0
                bar_id += 1
                
        return bar_ids


class DollarBar(BaseBar):
    """
    Dollar-volume based bars with fixed dollar volume per bar.
    
    Args:
        df: DataFrame with tick/trade data
        timestamp_col: Column name for timestamps
        price_col: Column name for prices
        volume_col: Column name for volumes
        interval_dollar: Target dollar volume per bar
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'ts_init',
        price_col: str = 'price',
        volume_col: str = 'size',
        interval_dollar: int = 100000,
    ):
        super().__init__(df, timestamp_col, price_col, volume_col, interval_dollar)
    
    def _get_group_idx(self) -> np.ndarray:
        return self._group_trades_by_dollar(
            self._df[self.volume_col].values,
            self._df[self.price_col].values,
            self.interval
        )
    
    @staticmethod
    @numba.jit(nopython=True)
    def _group_trades_by_dollar(
        df_qty: np.ndarray,
        df_price: np.ndarray,
        target_dollar: int
    ) -> np.ndarray:
        bar_ids = np.zeros(len(df_qty), dtype=np.int64)
        current_dollar = 0
        bar_id = 0
        
        for i in range(len(df_qty)):
            current_dollar += df_qty[i] * df_price[i]
            bar_ids[i] = bar_id
            if current_dollar >= target_dollar:
                current_dollar = 0
                bar_id += 1
                
        return bar_ids
