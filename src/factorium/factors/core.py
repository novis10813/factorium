import pandas as pd
import matplotlib.figure as mpl_figure

from typing import Union, Optional, List, Tuple, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

from .base import BaseFactor
from .mixins.math_ops import MathOpsMixin
from .mixins.ts_ops import TimeSeriesOpsMixin
from .mixins.cs_ops import CrossSectionalOpsMixin
from .plotting import FactorPlotter

if TYPE_CHECKING:
    from ..aggbar import AggBar


class Factor(CrossSectionalOpsMixin, TimeSeriesOpsMixin, MathOpsMixin, BaseFactor):
    """
    A factor representing a time-series of values for multiple symbols.
    
    Supports:
    - Arithmetic operations (+, -, *, /)
    - Comparison operations (<, <=, >, >=, ==, !=)
    - Time-series operations (ts_rank, ts_mean, ts_std, etc.)
    - Cross-sectional operations (rank, mean, median)
    - Mathematical operations (abs, log, pow, etc.)
    - Plotting operations (plot with various types)
    
    Example:
        >>> factor = Factor(data, name="close")
        >>> normalized = factor.ts_zscore(20)
        >>> ranked = normalized.rank()
        >>> ranked.plot(plot_type='timeseries')
    """
    
    def __init__(self, data: Union["AggBar", pd.DataFrame, Path], name: Optional[str] = None):
        super().__init__(data, name)
    
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
        Plot the factor data.
        
        Args:
            plot_type: Type of plot ('timeseries', 'heatmap', 'distribution')
            symbols: List of symbols to plot (None for all)
            start_time: Start time filter
            end_time: End time filter
            figsize: Figure size (width, height)
            **kwargs: Additional arguments passed to the plotter
            
        Returns:
            matplotlib Figure object
            
        Example:
            >>> factor.plot(plot_type='timeseries', symbols=['AAPL', 'MSFT'])
            >>> factor.plot(plot_type='heatmap', figsize=(14, 8))
            >>> factor.plot(plot_type='distribution', plot_type='histogram')
        """
        
        plotter = FactorPlotter(self)
        return plotter.plot(
            plot_type=plot_type,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            figsize=figsize,
            **kwargs
        )
