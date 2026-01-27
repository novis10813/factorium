"""Binance exchange adapter."""

from pathlib import Path

from .base import BaseExchangeAdapter, ColumnMapping


class BinanceAdapter(BaseExchangeAdapter):
    """Adapter for Binance exchange data.

    Supports:
    - Spot market: trades, aggTrades, klines
    - Futures UM (USDT-margined): trades, aggTrades, klines
    - Futures CM (Coin-margined): trades, aggTrades, klines
    """

    BINANCE_VISION_BASE = "https://data.binance.vision/data"

    @property
    def name(self) -> str:
        return "binance"

    @property
    def column_mappings(self) -> dict[str, ColumnMapping]:
        return {
            "aggTrades": ColumnMapping(
                timestamp="transact_time",
                price="price",
                volume="quantity",
                is_buyer_maker="is_buyer_maker",
            ),
            "trades": ColumnMapping(
                timestamp="time",
                price="price",
                volume="qty",
                is_buyer_maker="is_buyer_maker",
            ),
            "klines": ColumnMapping(
                timestamp="open_time",
                price="close",
                volume="volume",
                is_buyer_maker="",  # Not applicable for klines
            ),
        }

    def _get_market_string(self, market_type: str, futures_type: str = "") -> str:
        """Get market string for Hive partitioning."""
        if market_type == "futures":
            return f"futures_{futures_type}"
        return market_type

    def build_parquet_glob(
        self,
        base_path: Path,
        symbols: list[str],
        data_type: str,
        market_type: str,
        **kwargs,
    ) -> str:
        """Build glob pattern for Binance Parquet files.

        Uses Hive partitioning format:
        base_path/market=X/data_type=Y/symbol=Z/year=.../month=.../day=.../data.parquet
        """
        futures_type = kwargs.get("futures_type", "um")
        market_str = self._get_market_string(market_type, futures_type)

        if len(symbols) == 1:
            symbol_pattern = f"symbol={symbols[0]}"
        else:
            # For multiple symbols, we'll filter in SQL instead
            symbol_pattern = "symbol=*"

        return str(base_path / f"market={market_str}" / f"data_type={data_type}" / symbol_pattern / "**/*.parquet")

    def get_download_url(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        date: str,
        futures_type: str = "um",
        **kwargs,
    ) -> str:
        """Get Binance Vision download URL."""
        if market_type == "futures":
            market_path = f"futures/{futures_type}"
        else:
            market_path = "spot"

        return f"{self.BINANCE_VISION_BASE}/{market_path}/daily/{data_type}/{symbol}/{symbol}-{data_type}-{date}.zip"
