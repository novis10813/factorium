import pandas as pd
import matplotlib.figure as mpl_figure

from typing import Union, Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from pathlib import Path
from datetime import datetime

from .base import BaseFactor
from .mixins.math_ops import MathOpsMixin
from .mixins.ts_ops import TimeSeriesOpsMixin
from .mixins.cs_ops import CrossSectionalOpsMixin

if TYPE_CHECKING:
    from ..aggbar import AggBar
    from .analyzer import FactorAnalysisResult


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
        prices: Union["Factor", "AggBar"],
        periods: int = 1,  # MVP 僅支援單一窗口
        quantiles: int = 5,
        output_dir: Optional[str] = None,
        price_col: str = "close",
        **kwargs,
    ) -> "FactorAnalysisResult":
        """
        Evaluate factor's predictive power (Evaluation Layer).

        Args:
            prices: Price data (Factor or AggBar)
            periods: Prediction horizon (currently only supports single int)
            quantiles: Number of quantiles for layer analysis (default 5)
            output_dir: Experiment output directory (creates timestamped folder if specified)
            price_col: Price column name (default "close")

        Returns:
            FactorAnalysisResult: Complete evaluation metrics including IC, ICIR, t-stat,
                                  turnover, quantile returns, and cumulative returns

        Example:
            >>> momentum = ts_returns(close, 20)
            >>> result = momentum.eval(prices, output_dir="./experiments")
            >>> print(result.ic_summary)
            {'mean_ic': 0.05, 'ic_ir': 1.2, 't_stat': 3.5, ...}
        """
        from .analyzer import FactorAnalyzer

        analyzer = FactorAnalyzer(factor=self, prices=prices, quantiles=quantiles)

        result = analyzer.analyze(price_col=price_col, periods=periods)

        # Save results if output directory specified
        if output_dir:
            result.save(output_dir)

        return result

    def plot(
        self,
        plot_type: str = "timeseries",
        symbols: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        figsize: Tuple[int, int] = (12, 6),
        **kwargs,
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
            >>> factor.plot(plot_type='distribution', dist_type='histogram')
        """
        from .plotting import FactorPlotter

        plotter = FactorPlotter(self)
        return plotter.plot(
            plot_type=plot_type, symbols=symbols, start_time=start_time, end_time=end_time, figsize=figsize, **kwargs
        )
