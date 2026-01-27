from pathlib import Path
import pytest
from factorium.data.adapters.base import BaseExchangeAdapter, ColumnMapping
from factorium.data.adapters.binance import BinanceAdapter


class TestColumnMapping:
    def test_column_mapping_creation(self):
        """Test ColumnMapping can be created with required fields."""
        mapping = ColumnMapping(
            timestamp="transact_time",
            price="price",
            volume="quantity",
            is_buyer_maker="is_buyer_maker",
        )

        assert mapping.timestamp == "transact_time"
        assert mapping.price == "price"
        assert mapping.volume == "quantity"
        assert mapping.is_buyer_maker == "is_buyer_maker"


class TestBaseExchangeAdapter:
    def test_cannot_instantiate_directly(self):
        """Test that BaseExchangeAdapter cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseExchangeAdapter()

    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclass must implement all abstract methods."""

        class IncompleteAdapter(BaseExchangeAdapter):
            pass

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_complete_subclass_can_be_instantiated(self):
        """Test that a complete subclass can be instantiated."""

        class CompleteAdapter(BaseExchangeAdapter):
            @property
            def name(self) -> str:
                return "test"

            @property
            def column_mappings(self) -> dict:
                return {
                    "aggTrades": ColumnMapping(
                        timestamp="ts",
                        price="p",
                        volume="v",
                        is_buyer_maker="m",
                    )
                }

            def build_parquet_glob(self, *args, **kwargs) -> str:
                return "test/*.parquet"

            def get_download_url(self, *args, **kwargs) -> str:
                return "https://example.com"

        adapter = CompleteAdapter()
        assert adapter.name == "test"
        assert "aggTrades" in adapter.column_mappings

    def test_get_column_mapping_valid(self):
        """Test getting a valid column mapping."""

        class TestAdapter(BaseExchangeAdapter):
            @property
            def name(self) -> str:
                return "test"

            @property
            def column_mappings(self) -> dict:
                return {
                    "aggTrades": ColumnMapping(
                        timestamp="ts",
                        price="p",
                        volume="v",
                        is_buyer_maker="m",
                    )
                }

            def build_parquet_glob(self, *args, **kwargs) -> str:
                return "test/*.parquet"

            def get_download_url(self, *args, **kwargs) -> str:
                return "https://example.com"

        adapter = TestAdapter()
        mapping = adapter.get_column_mapping("aggTrades")

        assert mapping.timestamp == "ts"
        assert mapping.price == "p"

    def test_get_column_mapping_invalid_raises(self):
        """Test that invalid data type raises KeyError."""

        class TestAdapter(BaseExchangeAdapter):
            @property
            def name(self) -> str:
                return "test"

            @property
            def column_mappings(self) -> dict:
                return {}

            def build_parquet_glob(self, *args, **kwargs) -> str:
                return "test/*.parquet"

            def get_download_url(self, *args, **kwargs) -> str:
                return "https://example.com"

        adapter = TestAdapter()
        with pytest.raises(KeyError, match="Unsupported data type"):
            adapter.get_column_mapping("invalid_type")


class TestBinanceAdapter:
    """Tests for BinanceAdapter."""

    @pytest.fixture
    def adapter(self):
        return BinanceAdapter()

    def test_name(self, adapter):
        """Test adapter name is 'binance'."""
        assert adapter.name == "binance"

    def test_column_mappings_aggtrades(self, adapter):
        """Test column mapping for aggTrades."""
        mapping = adapter.get_column_mapping("aggTrades")

        assert mapping.timestamp == "transact_time"
        assert mapping.price == "price"
        assert mapping.volume == "quantity"
        assert mapping.is_buyer_maker == "is_buyer_maker"

    def test_column_mappings_trades(self, adapter):
        """Test column mapping for trades."""
        mapping = adapter.get_column_mapping("trades")

        assert mapping.timestamp == "time"
        assert mapping.price == "price"
        assert mapping.volume == "qty"
        assert mapping.is_buyer_maker == "is_buyer_maker"

    def test_build_parquet_glob_futures_um(self, adapter):
        """Test glob pattern for futures UM market."""
        pattern = adapter.build_parquet_glob(
            base_path=Path("/data"),
            symbols=["BTCUSDT"],
            data_type="aggTrades",
            market_type="futures",
            start_date="2024-01-01",
            end_date="2024-01-07",
            futures_type="um",
        )

        assert "market=futures_um" in pattern
        assert "data_type=aggTrades" in pattern
        assert "symbol=BTCUSDT" in pattern
        assert "**/*.parquet" in pattern

    def test_build_parquet_glob_spot(self, adapter):
        """Test glob pattern for spot market."""
        pattern = adapter.build_parquet_glob(
            base_path=Path("/data"),
            symbols=["BTCUSDT"],
            data_type="trades",
            market_type="spot",
            start_date="2024-01-01",
            end_date="2024-01-07",
        )

        assert "market=spot" in pattern
        assert "data_type=trades" in pattern

    def test_build_parquet_glob_multiple_symbols(self, adapter):
        """Test glob pattern for multiple symbols uses wildcard."""
        pattern = adapter.build_parquet_glob(
            base_path=Path("/data"),
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            data_type="aggTrades",
            market_type="futures",
            start_date="2024-01-01",
            end_date="2024-01-07",
            futures_type="um",
        )

        # Multiple symbols should use wildcard
        assert "symbol=*" in pattern

    def test_get_download_url_futures_um(self, adapter):
        """Test download URL for futures UM."""
        url = adapter.get_download_url(
            symbol="BTCUSDT",
            data_type="aggTrades",
            market_type="futures",
            date="2024-01-15",
            futures_type="um",
        )

        assert "data.binance.vision" in url
        assert "futures/um" in url
        assert "BTCUSDT" in url
        assert "aggTrades" in url
        assert "2024-01-15" in url
        assert ".zip" in url

    def test_unsupported_data_type_raises(self, adapter):
        """Test that unsupported data type raises KeyError."""
        with pytest.raises(KeyError, match="Unsupported data type"):
            adapter.get_column_mapping("invalid_type")
