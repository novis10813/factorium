import pandas as pd
import matplotlib.figure as mpl_figure

from typing import Union, Optional, List, Tuple, Dict, Any, TYPE_CHECKING
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
    
    @classmethod
    def from_expression(cls, expr: str, context: Dict[str, "Factor"]) -> "Factor":
        """
        Create a Factor from an expression string.
        
        Args:
            expr: Expression string using functional syntax (e.g., "ts_delta(close, 20) / ts_shift(close, 20)")
            context: Dictionary mapping variable names to Factor objects
            
        Returns:
            Factor: The resulting factor from the expression
            
        Example:
            >>> close = agg["close"]
            >>> momentum = Factor.from_expression(
            ...     "ts_delta(close, 20) / ts_shift(close, 20)",
            ...     context={'close': close}
            ... )
        """
        from .parser import FactorExpressionParser
        parser = FactorExpressionParser()
        return parser.parse(expr, context)
    
    def eval(
        self,
        prices: "Factor",
        periods: List[int] = (1, 5, 10),
        quantiles: int = 5,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a full evaluation report for the factor.
        
        Args:
            prices: A Factor object containing price data (e.g., close prices)
            periods: List of holding periods to evaluate (e.g., [1, 5, 10] days)
            quantiles: Number of quantiles for layer testing
            save_path: Path to save the evaluation report plot (e.g., 'report.png')
            **kwargs: Additional arguments passed to the evaluator
            
        Returns:
            Dictionary containing evaluation metrics:
            - ic_mean: Mean IC for each period
            - ic_ir: IC Information Ratio for each period
            - turnover_mean: Average factor turnover
            - layer_returns: Average returns for each quantile
            - spread: Long-short spread (Top - Bottom quantile)
            
        Example:
            >>> factor.eval(prices=close_factor, periods=[1, 5, 20], save_path='eval.png')
        """
        from .evaluation import FactorEvaluator
        evaluator = FactorEvaluator(self, prices)
        return evaluator.run_full_report(periods=periods, quantiles=quantiles, save_path=save_path, **kwargs)
    
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
