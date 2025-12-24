"""
Plotting utilities for Factor objects.

Provides FactorPlotter class for visualizing factor data with various plot types.
"""

from typing import Optional, List, Tuple, TYPE_CHECKING
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure

if TYPE_CHECKING:
    from .core import Factor


class FactorPlotter:
    """
    Plotting utility for Factor objects.
    
    Supports multiple plot types:
    - timeseries: Time series plot with one line per symbol
    - heatmap: Heatmap showing factor values across time and symbols
    - distribution: Distribution plots (histogram/density) of factor values
    """
    
    def __init__(self, factor: "Factor"):
        """
        Initialize FactorPlotter with a Factor object.
        
        Args:
            factor: Factor object to plot
        """
        self.factor = factor
        self.data = factor.data.copy()  # DataFrame with start_time, end_time, symbol, factor
        
    def _filter_data(
        self,
        symbols: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Filter data by symbols and time range.
        
        Args:
            symbols: List of symbols to include (None for all)
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Filtered DataFrame
        """
        filtered = self.data.copy()
        
        # Filter by symbols
        if symbols is not None:
            filtered = filtered[filtered['symbol'].isin(symbols)]
        
        # Filter by time range
        if start_time is not None:
            start_ts = int(pd.Timestamp(start_time).timestamp() * 1000)
            filtered = filtered[filtered['end_time'] >= start_ts]
        
        if end_time is not None:
            end_ts = int(pd.Timestamp(end_time).timestamp() * 1000)
            filtered = filtered[filtered['end_time'] <= end_ts]
        
        return filtered
    
    def plot_timeseries(
        self,
        symbols: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        figsize: Tuple[int, int] = (12, 6),
        **kwargs
    ) -> mpl_figure.Figure:
        """
        Plot time series of factor values for each symbol.
        
        Args:
            symbols: List of symbols to plot (None for all)
            start_time: Start time filter
            end_time: End time filter
            figsize: Figure size (width, height)
            **kwargs: Additional arguments passed to matplotlib
            
        Returns:
            matplotlib Figure object
        """
        filtered = self._filter_data(symbols, start_time, end_time)
        
        if filtered.empty:
            raise ValueError("No data to plot after filtering")
        
        # Convert timestamps to datetime
        filtered = filtered.copy()
        filtered['datetime'] = pd.to_datetime(filtered['end_time'], unit='ms')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each symbol
        for symbol in filtered['symbol'].unique():
            symbol_data = filtered[filtered['symbol'] == symbol].sort_values('datetime')
            ax.plot(symbol_data['datetime'], symbol_data['factor'], label=symbol, **kwargs)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Factor: {self.factor.name}')
        ax.set_title(f'Time Series: {self.factor.name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_heatmap(
        self,
        symbols: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        figsize: Tuple[int, int] = (14, 8),
        **kwargs
    ) -> mpl_figure.Figure:
        """
        Plot heatmap of factor values across time and symbols.
        
        Args:
            symbols: List of symbols to plot (None for all)
            start_time: Start time filter
            end_time: End time filter
            figsize: Figure size (width, height)
            **kwargs: Additional arguments passed to matplotlib
            
        Returns:
            matplotlib Figure object
        """
        filtered = self._filter_data(symbols, start_time, end_time)
        
        if filtered.empty:
            raise ValueError("No data to plot after filtering")
        
        # Convert timestamps to datetime
        filtered = filtered.copy()
        filtered['datetime'] = pd.to_datetime(filtered['end_time'], unit='ms')
        
        # Pivot to create matrix: symbols (rows) x time (columns)
        pivot_data = filtered.pivot_table(
            index='symbol',
            columns='datetime',
            values='factor',
            aggfunc='first'
        )
        
        # Sort symbols and time
        pivot_data = pivot_data.sort_index()
        pivot_data = pivot_data.sort_index(axis=1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(pivot_data.values, aspect='auto', cmap='RdYlGn_r', **kwargs)
        
        # Set ticks and labels
        n_symbols = len(pivot_data.index)
        n_times = len(pivot_data.columns)
        
        # Show subset of time labels to avoid crowding
        time_step = max(1, n_times // 10)
        time_indices = list(range(0, n_times, time_step))
        if n_times - 1 not in time_indices:
            time_indices.append(n_times - 1)
        
        ax.set_xticks(time_indices)
        ax.set_xticklabels([pivot_data.columns[i].strftime('%Y-%m-%d') for i in time_indices], rotation=45, ha='right')
        
        ax.set_yticks(range(n_symbols))
        ax.set_yticklabels(pivot_data.index)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Symbol')
        ax.set_title(f'Heatmap: {self.factor.name}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'Factor: {self.factor.name}')
        
        plt.tight_layout()
        return fig
    
    def plot_distribution(
        self,
        symbols: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        figsize: Tuple[int, int] = (12, 6),
        dist_type: str = 'histogram',
        **kwargs
    ) -> mpl_figure.Figure:
        """
        Plot distribution of factor values.
        
        Args:
            symbols: List of symbols to plot (None for all)
            start_time: Start time filter
            end_time: End time filter
            figsize: Figure size (width, height)
            dist_type: Type of distribution plot ('histogram' or 'density')
            **kwargs: Additional arguments passed to matplotlib
            
        Returns:
            matplotlib Figure object
        """
        filtered = self._filter_data(symbols, start_time, end_time)
        
        if filtered.empty:
            raise ValueError("No data to plot after filtering")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if dist_type == 'histogram':
            # Plot histogram for each symbol or combined
            if symbols is None or len(filtered['symbol'].unique()) > 10:
                # Too many symbols, plot combined distribution
                ax.hist(filtered['factor'].dropna(), bins=50, alpha=0.7, edgecolor='black', **kwargs)
                ax.set_xlabel(f'Factor: {self.factor.name}')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution: {self.factor.name} (All Symbols)')
            else:
                # Plot separate histograms for each symbol
                for symbol in filtered['symbol'].unique():
                    symbol_data = filtered[filtered['symbol'] == symbol]['factor'].dropna()
                    ax.hist(symbol_data, bins=30, alpha=0.6, label=symbol, **kwargs)
                ax.set_xlabel(f'Factor: {self.factor.name}')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution: {self.factor.name}')
                ax.legend()
        
        elif dist_type == 'density':
            # Plot density/KDE for each symbol or combined
            if symbols is None or len(filtered['symbol'].unique()) > 10:
                # Too many symbols, plot combined distribution
                filtered['factor'].dropna().plot.density(ax=ax, **kwargs)
                ax.set_xlabel(f'Factor: {self.factor.name}')
                ax.set_ylabel('Density')
                ax.set_title(f'Density: {self.factor.name} (All Symbols)')
            else:
                # Plot separate density plots for each symbol
                for symbol in filtered['symbol'].unique():
                    symbol_data = filtered[filtered['symbol'] == symbol]['factor'].dropna()
                    symbol_data.plot.density(ax=ax, label=symbol, **kwargs)
                ax.set_xlabel(f'Factor: {self.factor.name}')
                ax.set_ylabel('Density')
                ax.set_title(f'Density: {self.factor.name}')
                ax.legend()
        
        else:
            raise ValueError(f"Invalid dist_type: {dist_type}. Must be 'histogram' or 'density'")
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot(
        self,
        plot_type: str = 'timeseries',
        symbols: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        figsize: Tuple[int, int] = (12, 6),
        **kwargs
    ) -> mpl_figure.Figure:
        """
        Main plotting method that routes to specific plot types.
        
        Args:
            plot_type: Type of plot ('timeseries', 'heatmap', 'distribution')
            symbols: List of symbols to plot (None for all)
            start_time: Start time filter
            end_time: End time filter
            figsize: Figure size (width, height)
            **kwargs: Additional arguments passed to specific plot methods
            
        Returns:
            matplotlib Figure object
            
        Raises:
            ValueError: If plot_type is not supported
        """
        plot_type = plot_type.lower()
        
        if plot_type == 'timeseries':
            return self.plot_timeseries(symbols, start_time, end_time, figsize, **kwargs)
        elif plot_type == 'heatmap':
            return self.plot_heatmap(symbols, start_time, end_time, figsize, **kwargs)
        elif plot_type == 'distribution':
            return self.plot_distribution(symbols, start_time, end_time, figsize, **kwargs)
        else:
            raise ValueError(
                f"Unsupported plot_type: {plot_type}. "
                "Supported types: 'timeseries', 'heatmap', 'distribution'"
            )

