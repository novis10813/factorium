import pytest
import polars as pl

from factorium import AggBar
from factorium.factors import CompositeFactor


class TestCompositeFactor:
    """Tests for CompositeFactor multi-factor composition."""

    @pytest.fixture
    def sample_data(self):
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 10, 3600000))
        
        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH"]:
                base_price = {"BTC": 100.0, "ETH": 50.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append({
                    "start_time": ts,
                    "end_time": ts + 3600000,
                    "symbol": symbol,
                    "close": price,
                    "volume": 1000.0 * (1 + 0.02 * i),
                    "open": price, "high": price, "low": price,
                })
        
        return AggBar(pl.DataFrame(rows))

    def test_composite_with_equal_weights(self, sample_data):
        """CompositeFactor should combine factors with equal weights."""
        factor1 = sample_data["close"].cs_rank()
        factor2 = sample_data["volume"].cs_rank()
        
        composite = CompositeFactor([factor1, factor2])
        result = composite.to_factor()
        
        assert result.name == "composite"
        assert len(result.lazy.collect()) > 0

    def test_composite_with_custom_weights(self, sample_data):
        """CompositeFactor should support custom weights."""
        factor1 = sample_data["close"].cs_rank()
        factor2 = sample_data["volume"].cs_rank()
        
        composite = CompositeFactor(
            [factor1, factor2],
            weights=[0.7, 0.3],
            name="custom_combo",
        )
        result = composite.to_factor()
        
        assert result.name == "custom_combo"

    def test_composite_requires_matching_weights(self, sample_data):
        """Should raise error if weights don't match factors."""
        factor1 = sample_data["close"].cs_rank()
        factor2 = sample_data["volume"].cs_rank()
        
        with pytest.raises(ValueError, match="must match"):
            CompositeFactor([factor1, factor2], weights=[0.5])
