"""
ResearchSession provides a high-level API for factor research workflows.

Example:
    >>> from factorium.research import ResearchSession
    >>> session = ResearchSession.from_csv("data.csv")
    >>> signal = session.factor("close").cs_rank()
    >>> result = session.backtest(signal)
    >>> print(result.metrics)
"""

from typing import Optional, Union, Dict, Any
import polars as pl
import pandas as pd
from pathlib import Path

from ..aggbar import AggBar
from ..factors.core import Factor
from ..backtest.vectorized import VectorizedBacktester, BacktestResult


class ResearchSession:
    """
    High-level API for factor research workflows.

    Simplifies common operations: loading data, creating factors,
    running backtests, and analyzing results.

    Args:
        data: AggBar object containing OHLCV data
        default_frequency: Default rebalancing frequency for backtests
        default_initial_capital: Default initial capital for backtests
        default_transaction_cost: Default transaction cost rate

    Example:
        >>> session = ResearchSession(aggbar)
        >>> signal = session.factor("close").cs_rank()
        >>> result = session.backtest(signal, neutralization="market")
        >>> print(result.metrics["sharpe_ratio"])
    """

    def __init__(
        self,
        data: AggBar,
        default_frequency: str = "1h",
        default_initial_capital: float = 10000.0,
        default_transaction_cost: float = 0.0003,
    ):
        self.data = data
        self.default_frequency = default_frequency
        self.default_initial_capital = default_initial_capital
        self.default_transaction_cost = default_transaction_cost

    @classmethod
    def from_csv(cls, path: Union[str, Path], **kwargs) -> "ResearchSession":
        """Create ResearchSession from CSV file."""
        aggbar = AggBar.from_csv(Path(path))
        return cls(aggbar, **kwargs)

    @classmethod
    def from_parquet(cls, path: Union[str, Path], **kwargs) -> "ResearchSession":
        """Create ResearchSession from Parquet file."""
        df = pl.read_parquet(path)
        aggbar = AggBar.from_df(df)
        return cls(aggbar, **kwargs)

    @classmethod
    def from_df(cls, df: Union[pd.DataFrame, pl.DataFrame], **kwargs) -> "ResearchSession":
        """Create ResearchSession from DataFrame."""
        aggbar = AggBar.from_df(df)
        return cls(aggbar, **kwargs)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "ResearchSession":
        """
        Auto-detect format and load data.

        Supports: .csv, .parquet
        """
        path = Path(path)
        if path.suffix == ".csv":
            return cls.from_csv(path, **kwargs)
        elif path.suffix == ".parquet":
            return cls.from_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def analyze(
        self,
        factor: Factor,
        quantiles: int = 5,
    ) -> "FactorAnalysisResult":
        """
        Analyze factor using FactorAnalyzer.

        Args:
            factor: Factor to analyze
            quantiles: Number of quantiles for grouping

        Returns:
            FactorAnalysisResult with IC, autocorr, etc.

        Example:
            >>> signal = session.factor("close").cs_rank()
            >>> analysis = session.analyze(signal)
            >>> print(analysis.ic_summary)
        """
        from ..factors.analyzer import FactorAnalyzer

        analyzer = FactorAnalyzer(factor, self.data, quantiles=quantiles)
        return analyzer.analyze()

    def factor(self, column: str) -> Factor:
        """
        Create a Factor from a column in the data.

        Args:
            column: Column name (e.g., "close", "volume")

        Returns:
            Factor object for further transformations

        Example:
            >>> close = session.factor("close")
            >>> signal = close.cs_rank()
        """
        return self.data[column]

    def backtest(
        self,
        signal: Factor,
        neutralization: str = "market",
        entry_price: str = "close",
        frequency: Optional[str] = None,
        initial_capital: Optional[float] = None,
        transaction_cost: Optional[float] = None,
    ) -> BacktestResult:
        """
        Run backtest with given signal.

        Args:
            signal: Factor to use as trading signal
            neutralization: "market" for neutral, "none" for long-only
            entry_price: Price column to use for entries
            frequency: Rebalancing frequency (defaults to session default)
            initial_capital: Initial capital (defaults to session default)
            transaction_cost: Transaction cost rate (defaults to session default)

        Returns:
            BacktestResult with equity curve, metrics, etc.

        Example:
            >>> signal = session.factor("close").cs_rank()
            >>> result = session.backtest(signal)
            >>> print(result.metrics["sharpe_ratio"])
        """
        bt = VectorizedBacktester(
            prices=self.data,
            signal=signal,
            neutralization=neutralization,
            entry_price=entry_price,
            frequency=frequency or self.default_frequency,
            initial_capital=initial_capital or self.default_initial_capital,
            transaction_cost=transaction_cost or self.default_transaction_cost,
        )
        return bt.run()
