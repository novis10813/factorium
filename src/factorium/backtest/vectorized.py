"""Vectorized backtester using Polars for performance."""

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
import polars as pl

from ..aggbar import AggBar
from ..factors.core import Factor
from .metrics import calculate_metrics
from .utils import frequency_to_periods_per_year


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    equity_curve: pl.DataFrame  # columns: [end_time, total_value]
    returns: pl.DataFrame  # columns: [end_time, return]
    weights: pl.DataFrame  # columns: [end_time, symbol, weight]
    turnover: pl.DataFrame  # columns: [end_time, turnover, cost]
    metrics: dict[str, float]

    def to_pandas(self) -> "BacktestResultPandas":
        """Convert all DataFrames to pandas for backward compatibility."""
        return BacktestResultPandas(
            equity_curve=self.equity_curve.to_pandas(),
            returns=self.returns.to_pandas(),
            weights=self.weights.to_pandas(),
            turnover=self.turnover.to_pandas(),
            metrics=self.metrics,
        )


@dataclass
class BacktestResultPandas:
    """Pandas version of BacktestResult for backward compatibility."""

    equity_curve: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.DataFrame
    turnover: pd.DataFrame
    metrics: dict[str, float]


class VectorizedBacktester:
    """Vectorized backtester using Polars for high performance."""

    def __init__(
        self,
        prices: AggBar | pl.DataFrame,
        signal: Factor | pl.DataFrame,
        entry_price: str = "close",
        transaction_cost: float | tuple[float, float] = 0.0003,
        initial_capital: float = 10000.0,
        neutralization: Literal["market", "none"] = "market",
        frequency: str = "1h",
        constraints: list | None = None,
        mask: str | None = None,
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
        self._mask = mask

        # Convert inputs to Polars DataFrames
        if isinstance(prices, AggBar):
            if entry_price not in prices.cols:
                raise ValueError(f"entry_price '{entry_price}' not found in prices")
            if mask is not None and mask not in prices.cols:
                raise ValueError(f"mask '{mask}' not found in prices")
            self.prices_df = prices.to_polars()
        else:
            self.prices_df = prices
            if entry_price not in prices.columns:
                raise ValueError(f"entry_price '{entry_price}' not found in prices")
            if mask is not None and mask not in prices.columns:
                raise ValueError(f"mask '{mask}' not found in prices")

        if isinstance(signal, Factor):
            self.signal_df = signal.lazy.collect()
        else:
            self.signal_df = signal

        self._result: BacktestResult | None = None

    def run(self) -> BacktestResult:
        """
        Run the backtest and return results.

        Returns:
            BacktestResult with equity_curve, returns, weights, turnover, and metrics
        """
        # Step 1: Prepare data (prices, signals, asset returns)
        combined = self._prepare_data()

        # Step 2: Calculate weights (neutralization + constraints + renormalization)
        combined = self._calculate_weights(combined)

        # Step 3: Calculate per-symbol returns
        combined = self._calculate_returns(combined)

        # Step 4: Calculate equity and turnover
        equity_df, turnover_df = self._calculate_equity(combined)

        # Step 5: Extract final weights
        weights_df = combined.select(["end_time", "symbol", "weight"])

        # Step 6: Build result
        self._result = self._build_result(equity_df, turnover_df, weights_df)
        return self._result

    def summary(self) -> dict[str, Any]:
        """Return a summary of backtest results."""
        if self._result is None:
            raise RuntimeError("Must call run() before summary()")

        final_value = self._result.equity_curve["total_value"].to_list()[-1]

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_turnover": float(self._result.turnover["turnover"].sum()),
            "total_cost": float(self._result.turnover["cost"].sum()),
            **self._result.metrics,
        }

    def _prepare_data(self) -> pl.DataFrame:
        """Merge prices and signals, shift signals to avoid lookahead bias."""
        # Get the entry price column
        price_cols = ["end_time", "symbol", self.entry_price]
        if self._mask is not None:
            price_cols.append(self._mask)
        prices_df = self.prices_df.select(price_cols).rename({self.entry_price: "price"})

        # Prepare signal data
        signal_df = self.signal_df.select(["end_time", "symbol", "factor"]).rename({"factor": "signal"})

        # Join on end_time and symbol
        combined = prices_df.join(signal_df, on=["end_time", "symbol"], how="left")

        # Shift signal by 1 per symbol to use previous signal (avoid lookahead bias)
        combined = combined.with_columns([
            pl.col("signal").shift(1).over("symbol").alias("prev_signal")
        ]).drop("signal")

        # Calculate asset returns: r_t = price_t / price_{t-1} - 1
        combined = combined.with_columns([
            (pl.col("price") / pl.col("price").shift(1).over("symbol") - 1.0).alias("asset_return")
        ])

        # Sort for stable processing
        combined = combined.sort(["end_time", "symbol"])

        return combined

    def _calculate_weights(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate portfolio weights (cross-sectional)."""
        signal_col = "prev_signal"
        if self._mask is not None:
            signal_col = "_masked_signal"
            df = df.with_columns(
                pl.when(pl.col(self._mask).fill_null(False))
                .then(pl.col("prev_signal"))
                .otherwise(None)
                .alias(signal_col)
            )

        if self.neutralization == "market":
            # Market neutral: (signal - mean) / sum(|signal - mean|)
            from .utils import neutralize_weights_polars

            df = neutralize_weights_polars(df, signal_col, "end_time")
        else:  # long-only
            # Normalize positive signals to sum to 1
            positive_only = pl.when(pl.col(signal_col) > 0).then(pl.col(signal_col)).otherwise(0.0)
            df = df.with_columns(
                [(positive_only / positive_only.sum().over("end_time")).fill_nan(0.0).fill_null(0.0).alias("weight")]
            )

        if self._mask is not None:
            df = df.drop("_masked_signal")

        # Apply constraints
        for constraint in self.constraints:
            df = constraint.apply(df)

        # Restore weight invariants only after constraints (already normalized above)
        if self.constraints:
            from .utils import renormalize_weights

            df = renormalize_weights(df, neutralization=self.neutralization)

        return df

    def _calculate_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate portfolio returns from weights and asset returns."""
        # Previous weight per symbol (for turnover calculation)
        df = df.with_columns([
            pl.col("weight").shift(1).over("symbol").fill_null(0.0).alias("prev_weight")
        ])

        # Weight change per symbol
        df = df.with_columns([
            (pl.col("weight") - pl.col("prev_weight")).alias("weight_change")
        ])

        # Per-symbol contribution to portfolio return
        df = df.with_columns([
            (pl.col("weight") * pl.col("asset_return").fill_null(0.0)).alias("contribution")
        ])

        return df

    def _calculate_equity(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Calculate portfolio equity and turnover from per-symbol data.

        Returns:
            Tuple of (equity_df, turnover_df).
            equity_df columns: [end_time, total_value]
            turnover_df columns: [end_time, turnover, cost]
        """
        buy_rate, sell_rate = self.transaction_cost

        # Aggregate to time level
        per_period = (
            df.group_by("end_time")
            .agg([
                pl.col("contribution").sum().alias("gross_return"),
                # Buy-side turnover: sum of positive weight changes
                pl.col("weight_change").clip(lower_bound=0.0).sum().alias("buy_turnover"),
                # Sell-side turnover: sum of abs(negative weight changes)
                (-pl.col("weight_change").clip(upper_bound=0.0)).sum().alias("sell_turnover"),
            ])
            .sort("end_time")
        )

        # Calculate cost and net return
        per_period = per_period.with_columns([
            (pl.col("buy_turnover") + pl.col("sell_turnover")).alias("turnover"),
            (pl.col("buy_turnover") * buy_rate + pl.col("sell_turnover") * sell_rate).alias("cost"),
        ]).with_columns([
            (pl.col("gross_return") - pl.col("cost")).alias("net_return"),
        ])

        # Cumulative equity: equity_t = initial_capital * prod(1 + net_return)
        per_period = per_period.with_columns([
            (
                (1.0 + pl.col("net_return")).cum_prod() * self.initial_capital
            ).alias("total_value")
        ])

        equity_df = per_period.select(["end_time", "total_value"])
        turnover_df = per_period.select(["end_time", "turnover", "cost"])

        return equity_df, turnover_df

    def _calculate_metrics(self, equity_df: pl.DataFrame) -> dict[str, float]:
        """Calculate performance metrics by delegating to calculate_metrics()."""
        equity_pd = equity_df.to_pandas()
        returns_series = equity_pd["total_value"].pct_change().dropna()
        return calculate_metrics(
            returns_series,
            risk_free_rate=0.0,
            periods_per_year=self.periods_per_year,
        )

    def _build_result(
        self,
        equity_df: pl.DataFrame,
        turnover_df: pl.DataFrame,
        weights_df: pl.DataFrame,
    ) -> BacktestResult:
        """Assemble final result."""
        # Returns
        returns = (
            equity_df
            .with_columns([pl.col("total_value").pct_change().alias("return")])
            .select(["end_time", "return"])
        )

        # Calculate metrics
        metrics = self._calculate_metrics(equity_df)

        return BacktestResult(
            equity_curve=equity_df,
            returns=returns,
            weights=weights_df,
            turnover=turnover_df,
            metrics=metrics,
        )
