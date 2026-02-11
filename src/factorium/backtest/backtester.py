"""Factor-based backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

import warnings

import numpy as np
import pandas as pd

from .metrics import calculate_metrics
from .portfolio import Portfolio
from .utils import POSITION_EPSILON, frequency_to_periods_per_year, neutralize_weights, normalize_weights

if TYPE_CHECKING:
    import matplotlib.figure

    from ..aggbar import AggBar
    from ..factors.core import Factor


@dataclass
class IterativeBacktestResult:
    """Container for backtest results."""

    equity_curve: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]
    trades: pd.DataFrame
    portfolio_history: pd.DataFrame


class IterativeBacktester:
    """
    Factor-based backtesting engine (Iterative).

    Uses factor signals to generate position weights and simulates trading.

    Args:
        prices: Price data (AggBar)
        signal: Factor signal for position weighting
        entry_price: Column name for entry prices (default: "close")
        transaction_cost: Cost rate(s) as float or (buy, sell) tuple
        initial_capital: Starting capital (default: 10000.0)
        full_rebalance: If True, close all positions before rebalancing.
            WARNING: This doubles transaction costs as it sells all positions
            then re-buys target positions. Use only when complete portfolio
            reset is required.
        neutralization: "market" for dollar-neutral, "none" for long-only
        frequency: Trading frequency for annualization (e.g., "1h", "1d")

    Example:
        >>> from factorium import AggBar
        >>> from factorium.backtest import Backtester
        >>>
        >>> agg = AggBar.from_parquet("data.parquet")
        >>> signal = (agg["close"] / agg["close"].ts_shift(20) - 1).cs_rank()
        >>> bt = Backtester(prices=agg, signal=signal, neutralization="market")
        >>> result = bt.run()
        >>> print(bt.summary())
    """

    def __init__(
        self,
        prices: "AggBar",
        signal: "Factor",
        entry_price: str = "close",
        transaction_cost: Union[float, tuple[float, float]] = 0.0003,
        initial_capital: float = 10000.0,
        full_rebalance: bool = False,
        neutralization: Literal["market", "none"] = "market",
        frequency: str = "1h",
    ):
        warnings.warn(
            "IterativeBacktester is deprecated and will be removed in v2.0. "
            "Use VectorizedBacktester instead for better performance.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.prices = prices
        self.signal = signal
        self.entry_price = entry_price
        self.initial_capital = initial_capital
        self.full_rebalance = full_rebalance
        self.neutralization = neutralization
        self.frequency = frequency
        self._periods_per_year = frequency_to_periods_per_year(frequency)

        self._signal_df = self.signal.to_pandas()
        self._prices_df = self.prices.data

        if isinstance(transaction_cost, (int, float)):
            self.cost_rates = (float(transaction_cost), float(transaction_cost))
        else:
            self.cost_rates = transaction_cost

        self._validate_inputs()

        self._portfolio: Optional[Portfolio] = None
        self._result: Optional[IterativeBacktestResult] = None
        self._price_map: dict[Any, Any] = {}
        self._signal_map: dict[Any, Any] = {}

    def _validate_inputs(self) -> None:
        if self.entry_price not in self.prices.cols:
            raise ValueError(f"entry_price '{self.entry_price}' not found in prices")

        signal_times = set(self._signal_df["end_time"].unique())
        price_times = set(self._prices_df["end_time"].unique())
        common_times = signal_times & price_times

        if len(common_times) < 2:
            raise ValueError("signal and prices must have at least 2 common timestamps")

    def _get_common_timestamps(self) -> np.ndarray:
        signal_times = set(self._signal_df["end_time"].unique())
        price_times = set(self._prices_df["end_time"].unique())
        common = sorted(signal_times & price_times)
        return np.array(common)

    def _prepare_data_access(self) -> None:
        for t, group in self._prices_df.groupby("end_time"):
            self._price_map[int(t)] = group.set_index("symbol")[self.entry_price]  # type: ignore[arg-type]
        for t, group in self._signal_df.groupby("end_time"):
            self._signal_map[int(t)] = group.set_index("symbol")["factor"]  # type: ignore[arg-type]

    def _get_prices_at(self, timestamp: int) -> pd.Series:
        if not self._price_map:
            self._prepare_data_access()
        return self._price_map.get(timestamp, pd.Series(dtype=float))

    def _get_signal_at(self, timestamp: int) -> pd.Series:
        if not self._signal_map:
            self._prepare_data_access()
        return self._signal_map.get(timestamp, pd.Series(dtype=float))

    def _calculate_target_weights(self, signals: pd.Series) -> pd.Series:
        signals = signals.dropna()

        if len(signals) == 0:
            return pd.Series(dtype=float)

        if self.neutralization == "none":
            return normalize_weights(signals)
        elif self.neutralization == "market":
            return neutralize_weights(signals)
        else:
            raise ValueError(f"Unknown neutralization: {self.neutralization}")

    def _calculate_target_holdings(
        self,
        weights: pd.Series,
        prices: pd.Series,
        total_value: float,
    ) -> pd.Series:
        common_symbols = weights.index.intersection(prices.index)
        weights = weights.loc[common_symbols]
        prices = prices.loc[common_symbols]

        target_values = weights * total_value
        target_quantities = target_values / prices

        return target_quantities

    def _generate_orders(
        self,
        target_holdings: pd.Series,
        current_holdings: Dict[str, float],
    ) -> Dict[str, float]:
        orders: Dict[str, float] = {}
        all_symbols = set(target_holdings.index) | set(current_holdings.keys())

        for symbol in all_symbols:
            symbol_str = str(symbol)
            raw_target = target_holdings.get(symbol_str, 0.0)
            target = 0.0 if (raw_target is None or pd.isna(raw_target)) else float(raw_target)
            current = current_holdings.get(symbol_str, 0.0)
            diff = target - current

            if abs(diff) > POSITION_EPSILON:
                orders[symbol_str] = diff

        return orders

    def run(self) -> IterativeBacktestResult:
        self._portfolio = Portfolio(initial_capital=self.initial_capital)

        timestamps = self._get_common_timestamps()

        first_prices = self._get_prices_at(timestamps[0])
        self._portfolio.record_snapshot(timestamps[0], first_prices)

        for i in range(1, len(timestamps)):
            current_ts = timestamps[i]
            prev_ts = timestamps[i - 1]

            current_prices = self._get_prices_at(current_ts)
            prev_signal = self._get_signal_at(prev_ts)

            if prev_signal.dropna().empty:
                self._portfolio.record_snapshot(current_ts, current_prices)
                continue

            target_weights = self._calculate_target_weights(prev_signal)

            if target_weights.empty:
                self._portfolio.record_snapshot(current_ts, current_prices)
                continue

            total_value = self._portfolio.get_total_value(current_prices)
            target_holdings = self._calculate_target_holdings(target_weights, current_prices, total_value)

            if self.full_rebalance:
                for symbol, qty in list(self._portfolio.positions.items()):
                    if symbol in current_prices.index:
                        self._portfolio.execute_trade(
                            symbol,
                            -qty,
                            float(current_prices[symbol]),
                            self.cost_rates,
                            int(current_ts),
                        )

            orders = self._generate_orders(target_holdings, self._portfolio.positions)

            for symbol, qty in orders.items():
                if symbol in current_prices.index:
                    self._portfolio.execute_trade(
                        symbol,
                        qty,
                        float(current_prices[symbol]),
                        self.cost_rates,
                        int(current_ts),
                    )

            self._portfolio.record_snapshot(current_ts, current_prices)

        return self._build_result()

    def _build_result(self) -> IterativeBacktestResult:
        assert self._portfolio is not None

        history_df = self._portfolio.get_history_df()
        trades_df = self._portfolio.get_trade_log_df()

        equity_curve = pd.Series(
            history_df["total_value"].values,
            index=pd.to_datetime(history_df["timestamp"], unit="ms"),
            name="equity",
        )

        returns = equity_curve.pct_change().dropna()
        returns.name = "returns"

        metrics = calculate_metrics(returns, periods_per_year=self._periods_per_year)

        self._result = IterativeBacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics,
            trades=trades_df,
            portfolio_history=history_df,
        )

        return self._result

    def summary(self) -> Dict[str, Any]:
        if self._result is None:
            raise RuntimeError("Must call run() before summary()")

        return {
            "initial_capital": self.initial_capital,
            "final_value": self._result.equity_curve.iloc[-1],
            "num_trades": len(self._result.trades),
            **self._result.metrics,
        }

    def plot_equity(self, figsize: tuple[float, float] = (12, 6)) -> "matplotlib.figure.Figure":
        import matplotlib.pyplot as plt

        if self._result is None:
            raise RuntimeError("Must call run() before plot_equity()")

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        self._result.equity_curve.plot(ax=axes[0], title="Equity Curve")
        axes[0].set_ylabel("Portfolio Value")
        axes[0].grid(True, alpha=0.3)

        rolling_max = self._result.equity_curve.cummax()
        drawdown = (self._result.equity_curve - rolling_max) / rolling_max
        drawdown.plot(ax=axes[1], title="Drawdown", color="red")
        axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @property
    def equity_curve(self) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Must call run() before accessing equity_curve")
        return self._result.equity_curve

    @property
    def returns(self) -> pd.Series:
        if self._result is None:
            raise RuntimeError("Must call run() before accessing returns")
        return self._result.returns
