"""Base class for exchange adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ColumnMapping:
    """Column name mapping for exchange-specific data formats.

    Attributes:
        timestamp: Column name for transaction timestamp (milliseconds)
        price: Column name for trade price
        volume: Column name for trade volume/quantity
        is_buyer_maker: Column name for buyer/seller indicator
    """

    timestamp: str
    price: str
    volume: str
    is_buyer_maker: str


class BaseExchangeAdapter(ABC):
    """Abstract base class for exchange data adapters.

    Provides a unified interface for:
    - Column name mapping across different exchanges
    - Parquet file path construction
    - Download URL generation

    Subclasses must implement all abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return exchange name (e.g., 'binance', 'bybit')."""
        ...

    @property
    @abstractmethod
    def column_mappings(self) -> dict[str, ColumnMapping]:
        """Return column mappings for each data type.

        Returns:
            Dict mapping data_type (e.g., 'aggTrades') to ColumnMapping
        """
        ...

    @abstractmethod
    def build_parquet_glob(
        self,
        base_path: Path,
        symbols: list[str],
        data_type: str,
        market_type: str,
        **kwargs,
    ) -> str:
        """Build glob pattern for Parquet files."""
        ...

    @abstractmethod
    def get_download_url(
        self,
        symbol: str,
        data_type: str,
        market_type: str,
        date: str,
        **kwargs,
    ) -> str:
        """Get download URL for a specific data file."""
        ...

    def get_column_mapping(self, data_type: str) -> ColumnMapping:
        """Get column mapping for a specific data type.

        Raises:
            KeyError: If data_type is not supported
        """
        if data_type not in self.column_mappings:
            raise KeyError(f"Unsupported data type: {data_type}. Available: {list(self.column_mappings.keys())}")
        return self.column_mappings[data_type]
