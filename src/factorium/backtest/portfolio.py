"""Portfolio management for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .utils import POSITION_EPSILON


@dataclass
class Portfolio:
    """Manages positions, cash, and trade execution."""

    initial_capital: float = 10000.0
    cash: float = field(init=False)
    positions: Dict[str, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    trade_log: List[Dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital

    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        cost_rates: tuple[float, float],
        timestamp: int,
    ) -> None:
        if abs(quantity) < POSITION_EPSILON:
            return

        trade_value = abs(quantity * price)
        is_buy = quantity > 0
        cost_rate = cost_rates[0] if is_buy else cost_rates[1]
        cost = trade_value * cost_rate

        if is_buy:
            self.cash -= trade_value + cost
        else:
            self.cash += trade_value - cost

        current_qty = self.positions.get(symbol, 0.0)
        new_qty = current_qty + quantity

        if abs(new_qty) < POSITION_EPSILON:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty

        self.trade_log.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "cost": cost,
                "trade_value": trade_value,
            }
        )

    def get_market_value(self, prices: pd.Series) -> float:
        market_value = 0.0
        for symbol, qty in self.positions.items():
            if symbol in prices.index:
                market_value += qty * prices[symbol]
        return market_value

    def get_total_value(self, prices: pd.Series) -> float:
        return self.cash + self.get_market_value(prices)

    def record_snapshot(self, timestamp: int, prices: pd.Series) -> None:
        self.history.append(
            {
                "timestamp": timestamp,
                "cash": self.cash,
                "market_value": self.get_market_value(prices),
                "total_value": self.get_total_value(prices),
                "num_positions": len(self.positions),
            }
        )

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def get_trade_log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)
