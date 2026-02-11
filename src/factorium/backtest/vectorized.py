"""Vectorized backtester using Polars for performance."""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Union
import numpy as np
import pandas as pd
import polars as pl

from ..aggbar import AggBar
from ..factors.core import Factor
from ..constants import EPSILON
from .utils import frequency_to_periods_per_year
from .metrics import calculate_metrics


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    equity_curve: pl.DataFrame  # columns: [end_time, total_value]
    returns: pl.DataFrame  # columns: [end_time, return]
    metrics: Dict[str, float]
    trades: pl.DataFrame  # columns: [end_time, symbol, qty, price, cost]
    portfolio_history: pl.DataFrame  # columns: [end_time, cash, market_value, total_value]

    def to_pandas(self) -> "BacktestResultPandas":
        """Convert all DataFrames to pandas for backward compatibility."""
        return BacktestResultPandas(
            equity_curve=self.equity_curve.to_pandas(),
            returns=self.returns.to_pandas(),
            metrics=self.metrics,
            trades=self.trades.to_pandas(),
            portfolio_history=self.portfolio_history.to_pandas(),
        )


@dataclass
class BacktestResultPandas:
    """Pandas version of BacktestResult for backward compatibility."""

    equity_curve: pd.DataFrame
    returns: pd.DataFrame
    metrics: Dict[str, float]
    trades: pd.DataFrame
    portfolio_history: pd.DataFrame


class VectorizedBacktester:
    """Vectorized backtester using Polars for high performance."""

    def __init__(
        self,
        prices: Union[AggBar, pl.DataFrame],
        signal: Union[Factor, pl.DataFrame],
        entry_price: str = "close",
        transaction_cost: Union[float, tuple[float, float]] = 0.0003,
        initial_capital: float = 10000.0,
        neutralization: Literal["market", "none"] = "market",
        frequency: str = "1h",
        constraints: Optional[list] = None,
    ):
        """
        Initialize the vectorized backtester.

        Args:
            prices: AggBar or Polars DataFrame with OHLCV data
            signal: Factor or Polars DataFrame with signals
            entry_price: Column name in prices for execution price
            transaction_cost: Transaction cost as % of notional, or (buy, sell) tuple
            initial_capital: Starting portfolio value
            neutralization: "market" for market neutral, "none" for long-only
            frequency: Frequency string (e.g., "1h", "1d")
            constraints: List of WeightConstraint objects to apply
        """
        self.initial_capital = initial_capital

        # Normalize transaction cost
        if isinstance(transaction_cost, (int, float)):
            self.transaction_cost = (float(transaction_cost), float(transaction_cost))
        else:
            self.transaction_cost = transaction_cost

        self.entry_price = entry_price
        self.neutralization = neutralization
        self.frequency = frequency
        self.periods_per_year = frequency_to_periods_per_year(frequency)
        self._periods_per_year = self.periods_per_year  # Alias for backward compatibility
        self.constraints = constraints or []

        # Convert inputs to Polars DataFrames
        if isinstance(prices, AggBar):
            if entry_price not in prices.cols:
                raise ValueError(f"entry_price '{entry_price}' not found in prices")
            self.prices_df = prices.to_polars()
        else:
            self.prices_df = prices
            if entry_price not in prices.columns:
                raise ValueError(f"entry_price '{entry_price}' not found in prices")

        if isinstance(signal, Factor):
            self.signal_df = signal.lazy.collect()
        else:
            self.signal_df = signal

        self._result: Optional[BacktestResult] = None

    def run(self) -> BacktestResult:
        """
        Run the backtest and return results.

        Returns:
            BacktestResult with equity_curve, returns, and metrics
        """
        # Step 1: Prepare data
        combined = self._prepare_data()

        # Step 2: Calculate weights
        combined = self._calculate_weights(combined)

        # Step 3: Calculate positions
        combined = self._calculate_positions(combined)

        # Step 4: Calculate equity
        portfolio_history = self._calculate_equity(combined)

        # Step 5: Build result
        self._result = self._build_result(portfolio_history, combined)
        return self._result

    def summary(self) -> Dict[str, Any]:
        """Return a summary of backtest results."""
        if self._result is None:
            raise RuntimeError("Must call run() before summary()")

        final_value = self._result.equity_curve["total_value"].to_list()[-1]

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "num_trades": len(self._result.trades),
            **self._result.metrics,
        }

    def _prepare_data(self) -> pl.DataFrame:
        """Merge prices and signals, shift signals to avoid lookahead bias."""
        # Get the entry price column
        prices_df = self.prices_df.select(["end_time", "symbol", self.entry_price]).rename({self.entry_price: "price"})

        # Prepare signal data
        signal_df = self.signal_df.select(["end_time", "symbol", "factor"]).rename({"factor": "signal"})

        # Join on end_time and symbol
        combined = prices_df.join(signal_df, on=["end_time", "symbol"], how="left")

        # Shift signal by 1 per symbol to use previous signal (avoid lookahead bias)
        combined = combined.with_columns([pl.col("signal").shift(1).over("symbol").alias("prev_signal")]).drop("signal")

        # Sort for stable processing
        combined = combined.sort(["end_time", "symbol"])

        return combined

    def _calculate_weights(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate portfolio weights (cross-sectional)."""
        if self.neutralization == "market":
            # Market neutral: (signal - mean) / sum(|signal - mean|)
            from .utils import neutralize_weights_polars

            df = neutralize_weights_polars(df, "prev_signal", "end_time")
        else:  # long-only
            # Normalize positive signals to sum to 1
            positive_only = pl.when(pl.col("prev_signal") > 0).then(pl.col("prev_signal")).otherwise(0.0)
            df = df.with_columns(
                [(positive_only / positive_only.sum().over("end_time")).fill_nan(0.0).fill_null(0.0).alias("weight")]
            )

        # Apply constraints
        for constraint in self.constraints:
            df = constraint.apply(df)

        return df

    def _calculate_positions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate target quantities and trades."""
        # Target quantity: weight * capital / price
        df = df.with_columns([(pl.col("weight") * self.initial_capital / pl.col("price")).alias("target_qty")])

        # Previous quantity (from previous time period)
        df = df.with_columns([pl.col("target_qty").shift(1).over("symbol").fill_null(0.0).alias("prev_qty")])

        # Trade quantity
        df = df.with_columns([(pl.col("target_qty") - pl.col("prev_qty")).alias("trade_qty")])

        # Trade cost
        buy_rate, sell_rate = self.transaction_cost
        df = df.with_columns(
            [
                pl.when(pl.col("trade_qty") > 0)
                .then(pl.col("trade_qty") * pl.col("price") * buy_rate)
                .otherwise(pl.col("trade_qty").abs() * pl.col("price") * sell_rate)
                .alias("trade_cost")
            ]
        )

        return df

    def _calculate_equity(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate portfolio equity over time."""
        # Aggregate to time level
        equity = (
            df.group_by("end_time")
            .agg(
                [
                    # Market value of holdings
                    (pl.col("target_qty") * pl.col("price")).sum().alias("market_value"),
                    # Total transaction costs
                    pl.col("trade_cost").sum().alias("total_trade_cost"),
                    # Net buy amount (negative = sold)
                    (pl.col("trade_qty") * pl.col("price")).sum().alias("net_buy"),
                ]
            )
            .sort("end_time")
        )

        # Cumulative calculations
        equity = (
            equity.with_columns(
                [
                    # Cumulative costs
                    pl.col("total_trade_cost").cum_sum().alias("cumulative_costs"),
                    # Cumulative net buys
                    pl.col("net_buy").cum_sum().alias("cumulative_buys"),
                ]
            )
            .with_columns(
                [
                    # Cash: capital - costs - net buys
                    (self.initial_capital - pl.col("cumulative_costs") - pl.col("cumulative_buys")).alias("cash"),
                ]
            )
            .with_columns(
                [
                    # Total value: cash + market value
                    (pl.col("cash") + pl.col("market_value")).alias("total_value")
                ]
            )
        )

        return equity.select(["end_time", "cash", "market_value", "total_value"])

    def _calculate_metrics(self, equity_history: pl.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Convert to pandas for metrics calculation
        equity_pd = equity_history.to_pandas()

        # Calculate period returns
        equity_pd["return"] = equity_pd["total_value"].pct_change()

        # Calculate metrics
        returns_series = equity_pd["return"].dropna()

        if len(returns_series) < 2:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        metrics = calculate_metrics(
            returns_series,
            risk_free_rate=0.0,
            periods_per_year=self.periods_per_year,
        )

        # Ensure Sortino, Calmar, and win rate follow specific requirements
        # Sortino ratio: annual_return / downside_std
        downside_returns = returns_series[returns_series < 0]
        if len(downside_returns) > 0:
            downside_std = float(downside_returns.std() * (self.periods_per_year**0.5))
            if downside_std > EPSILON:
                metrics["sortino_ratio"] = metrics["annual_return"] / downside_std
            else:
                metrics["sortino_ratio"] = np.inf if metrics["annual_return"] > 0 else 0.0
        else:
            metrics["sortino_ratio"] = np.inf if metrics["annual_return"] > 0 else 0.0

        # Calmar ratio: annual_return / abs(max_drawdown)
        max_dd = abs(metrics.get("max_drawdown", 0.0))
        if max_dd > EPSILON:
            metrics["calmar_ratio"] = metrics["annual_return"] / max_dd
        else:
            metrics["calmar_ratio"] = 0.0

        # Win rate: (returns > 0).sum() / len(returns)
        metrics["win_rate"] = float((returns_series > 0).sum()) / len(returns_series)

        return metrics

    def _build_result(
        self,
        portfolio_history: pl.DataFrame,
        combined_df: pl.DataFrame,
    ) -> BacktestResult:
        """Assemble final result."""
        # Equity curve: just end_time and total_value
        equity_curve = portfolio_history.select(["end_time", "total_value"])

        # Returns
        returns = (
            portfolio_history.select(["end_time", "total_value"])
            .with_columns([pl.col("total_value").pct_change().alias("return")])
            .select(["end_time", "return"])
        )

        # Trades
        trades = (
            combined_df.select(["end_time", "symbol", "trade_qty", "price", "trade_cost"])
            .rename({"trade_qty": "qty"})
            .filter(pl.col("qty") != 0)
        )

        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_history)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics,
            trades=trades,
            portfolio_history=portfolio_history,
        )
