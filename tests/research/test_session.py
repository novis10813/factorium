import pytest
import polars as pl

from factorium import AggBar
from factorium.research import ResearchSession


class TestResearchSessionInit:
    """Tests for ResearchSession initialization."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        timestamps = list(range(1704067200000, 1704067200000 + 3600000 * 30, 3600000))

        rows = []
        for i, ts in enumerate(timestamps):
            for symbol in ["BTC", "ETH", "SOL"]:
                base_price = {"BTC": 100.0, "ETH": 50.0, "SOL": 10.0}[symbol]
                price = base_price * (1 + 0.01 * i)
                rows.append(
                    {
                        "start_time": ts,
                        "end_time": ts + 3600000,
                        "symbol": symbol,
                        "open": price * 0.99,
                        "high": price * 1.01,
                        "low": price * 0.98,
                        "close": price,
                        "volume": 1000.0,
                    }
                )

        return AggBar(pl.DataFrame(rows))

    def test_init_with_aggbar(self, sample_data):
        """Should initialize with AggBar."""
        session = ResearchSession(sample_data)
        assert session.data is not None
        assert len(session.data.symbols) == 3

    def test_factor_creates_factor_object(self, sample_data):
        """session.factor() should return Factor."""
        session = ResearchSession(sample_data)
        close_factor = session.factor("close")

        from factorium.factors import Factor

        assert isinstance(close_factor, Factor)
        assert close_factor.name == "close"

    def test_backtest_returns_result(self, sample_data):
        """session.backtest() should return BacktestResult."""
        session = ResearchSession(sample_data)
        signal = session.factor("close").cs_rank()

        result = session.backtest(signal)

        from factorium.backtest.vectorized import BacktestResult

        assert isinstance(result, BacktestResult)
        assert result.metrics is not None
