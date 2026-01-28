"""
ResearchSession provides a high-level API for factor research workflows.

Example:
    >>> from factorium.research import ResearchSession
    >>> session = ResearchSession.from_csv("data.csv")
    >>> signal = session.factor("close").cs_rank()
    >>> result = session.backtest(signal)
    >>> print(result.metrics)
"""

from typing import Optional, Union, Dict, Any, List, Callable
import polars as pl
import pandas as pd
from pathlib import Path

from ..aggbar import AggBar
from ..factors.core import Factor
from ..factors.parser import FactorExpressionParser
from ..backtest.vectorized import VectorizedBacktester, BacktestResult


class ResearchSession:
    """
    High-level API for factor research workflows.

    Simplifies common operations: loading data, creating factors,
    running backtests, and analyzing results.

    Args:
        data: AggBar object or DataFrame containing OHLCV data
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
        data: Union[AggBar, pd.DataFrame, pl.DataFrame],
        default_frequency: str = "1h",
        default_initial_capital: float = 10000.0,
        default_transaction_cost: float = 0.0003,
    ):
        # Convert DataFrame to AggBar if needed
        if isinstance(data, (pd.DataFrame, pl.DataFrame)):
            data = AggBar.from_df(data)

        self.data = data
        self.default_frequency = default_frequency
        self.default_initial_capital = default_initial_capital
        self.default_transaction_cost = default_transaction_cost
        self._factors: Dict[str, Factor] = {}  # Cache for created factors
        self._parser = FactorExpressionParser()

    @property
    def symbols(self) -> List[str]:
        """Return list of symbols in the data."""
        return self.data.symbols

    @property
    def cols(self) -> List[str]:
        """Return list of columns in the data."""
        return self.data.cols

    def create_factor(self, expr: Union[str, Callable[[AggBar], Factor]], name: Optional[str] = None) -> Factor:
        """
        Create and cache a factor from expression or callable.

        Args:
            expr: Column name (str), expression string (str), or callable that takes AggBar
            name: Optional name for the factor

        Returns:
            Cached Factor object

        Example:
            >>> session.create_factor("close", "price")
            >>> session.create_factor("ts_mean(close, 20)", "ma20")
            >>> session.create_factor(lambda agg: agg["close"].ts_return(20), "ret_20d")
        """
        # Generate cache key
        cache_key = name or (expr if isinstance(expr, str) else id(expr))

        # Return cached if exists
        if cache_key in self._factors:
            return self._factors[cache_key]

        # Create new factor
        factor: Factor
        if isinstance(expr, str):
            if expr in self.data.cols:
                res = self.data[expr]
                if not isinstance(res, Factor):
                    raise TypeError(f"Expected Factor for column {expr}, got {type(res)}")
                factor = res
            else:
                # Build context for parser
                context = {
                    col: self.data[col] for col in self.data.cols if col not in ["start_time", "end_time", "symbol"]
                }
                factor = self._parser.parse(expr, context)

            if name:
                factor.name = name

        elif callable(expr):
            factor = expr(self.data)
            if name:
                factor.name = name
        else:
            raise TypeError(f"expr must be str or callable, got {type(expr)}")

        # Cache and return
        self._factors[cache_key] = factor
        return factor

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
        price_col: str = "close",
        quantiles: int = 5,
        periods: int = 1,
    ) -> "FactorAnalysisResult":
        """
        Analyze factor using FactorAnalyzer.

        Args:
            factor: Factor to analyze
            price_col: Price column for return calculation
            quantiles: Number of quantiles for grouping
            periods: Forward return periods for analysis

        Returns:
            FactorAnalysisResult dataclass with IC and quantile analysis
        """
        from ..factors.analyzer import FactorAnalyzer

        analyzer = FactorAnalyzer(factor, self.data, quantiles=quantiles)
        return analyzer.analyze(price_col=price_col, periods=periods)

    def quick_report(
        self,
        factor: Factor,
        periods: int = 1,
        quantiles: int = 5,
        price_col: str = "close",
    ) -> str:
        """
        Generate quick text summary of factor analysis and backtest.

        Args:
            factor: Factor to analyze and backtest
            periods: Forward return periods for analysis
            quantiles: Number of quantiles
            price_col: Price column for calculations

        Returns:
            Formatted text report

        Example:
            >>> session = ResearchSession(data)
            >>> signal = session.factor("close").cs_rank()
            >>> print(session.quick_report(signal))
        """
        # Run analysis
        analysis = self.analyze(factor, price_col=price_col, quantiles=quantiles, periods=periods)

        # Run backtest
        backtest = self.backtest(factor)

        # Format report
        ic_summary = analysis.ic_summary
        metrics = backtest.metrics

        report = f"""
Factor Analysis Report: {factor.name}
{"=" * 60}

IC Analysis (periods={periods}):
  Mean IC:        {ic_summary.get("mean_ic", 0):.4f}
  IC Std:         {ic_summary.get("ic_std", 0):.4f}
  IC IR:          {ic_summary.get("ic_ir", 0):.4f}

Backtest Performance:
  Total Return:   {metrics.get("total_return", 0):.2%}
  Annual Return:  {metrics.get("annual_return", 0):.2%}
  Sharpe Ratio:   {metrics.get("sharpe_ratio", 0):.2f}
  Max Drawdown:   {metrics.get("max_drawdown", 0):.2%}
  Sortino Ratio:  {metrics.get("sortino_ratio", 0):.2f}
  Calmar Ratio:   {metrics.get("calmar_ratio", 0):.2f}
  Win Rate:       {metrics.get("win_rate", 0):.2%}

Symbols: {len(self.symbols)}
Period: {self.data.timestamps.min()} to {self.data.timestamps.max()}
{"=" * 60}
"""
        return report.strip()

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
        res = self.data[column]
        if not isinstance(res, Factor):
            raise TypeError(f"Expected Factor, got {type(res)}")
        return res

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
        # Ensure neutralization is of correct type for VectorizedBacktester
        # which expects Literal["market", "none"]
        if neutralization not in ["market", "none"]:
            raise ValueError(f"neutralization must be 'market' or 'none', got {neutralization}")

        bt = VectorizedBacktester(
            prices=self.data,
            signal=signal,
            neutralization=neutralization,  # type: ignore
            entry_price=entry_price,
            frequency=frequency or self.default_frequency,
            initial_capital=initial_capital or self.default_initial_capital,
            transaction_cost=transaction_cost or self.default_transaction_cost,
        )
        return bt.run()

    def slice(
        self,
        start: Optional[Union[int, str]] = None,
        end: Optional[Union[int, str]] = None,
        symbols: Optional[List[str]] = None,
    ) -> "ResearchSession":
        """
        Create new session with subset of data.

        Args:
            start: Start timestamp (ms or ISO string)
            end: End timestamp (ms or ISO string)
            symbols: Symbol list to include

        Returns:
            New ResearchSession with filtered data
        """
        import pandas as pd

        df = self.data.to_polars()

        # Time filters
        if start is not None:
            if isinstance(start, str):
                start = int(pd.Timestamp(start).value // 1_000_000)
            df = df.filter(pl.col("end_time") >= start)

        if end is not None:
            if isinstance(end, str):
                end = int(pd.Timestamp(end).value // 1_000_000)
            df = df.filter(pl.col("end_time") <= end)

        # Symbol filter
        if symbols is not None:
            df = df.filter(pl.col("symbol").is_in(symbols))

        # Create new session
        from ..aggbar import AggBar

        new_aggbar = AggBar(df)

        return ResearchSession(
            new_aggbar,
            default_frequency=self.default_frequency,
            default_initial_capital=self.default_initial_capital,
            default_transaction_cost=self.default_transaction_cost,
        )
