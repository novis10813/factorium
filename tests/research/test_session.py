import pytest
import polars as pl

from factorium import AggBar
from factorium.research import ResearchSession


class TestResearchSessionMethods:
    """Tests for ResearchSession analysis and loading methods."""

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

    def test_analyze_returns_result(self, sample_data):
        """analyze() should return FactorAnalysisResult."""
        session = ResearchSession(sample_data)
        signal = session.factor("close").cs_rank()

        result = session.analyze(signal)

        # FactorAnalyzer returns dict, not FactorAnalysisResult yet
        assert result is not None
        assert "ic_summary" in result
        assert "quantile_returns" in result

    def test_from_df_pandas(self):
        """from_df() should work with pandas DataFrame."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "start_time": [1704067200000, 1704070800000],
                "end_time": [1704070800000, 1704074400000],
                "symbol": ["BTC", "BTC"],
                "close": [100.0, 101.0],
                "open": [99.0, 100.0],
                "high": [101.0, 102.0],
                "low": [98.0, 99.0],
                "volume": [1000.0, 1100.0],
            }
        )

        session = ResearchSession.from_df(df)
        assert len(session.data.symbols) == 1

    def test_from_df_polars(self):
        """from_df() should work with Polars DataFrame."""
        df = pl.DataFrame(
            {
                "start_time": [1704067200000, 1704070800000],
                "end_time": [1704070800000, 1704074400000],
                "symbol": ["BTC", "BTC"],
                "close": [100.0, 101.0],
                "open": [99.0, 100.0],
                "high": [101.0, 102.0],
                "low": [98.0, 99.0],
                "volume": [1000.0, 1100.0],
            }
        )

        session = ResearchSession.from_df(df)
        assert len(session.data.symbols) == 1

    def test_load_autodetects_format(self, tmp_path):
        """load() should auto-detect CSV and Parquet formats."""
        df = pl.DataFrame(
            {
                "start_time": [1704067200000],
                "end_time": [1704070800000],
                "symbol": ["BTC"],
                "close": [100.0],
                "open": [99.0],
                "high": [101.0],
                "low": [98.0],
                "volume": [1000.0],
            }
        )

        # Test CSV
        csv_path = tmp_path / "data.csv"
        df.to_pandas().to_csv(csv_path, index=False)
        session = ResearchSession.load(csv_path)
        assert session is not None

        # Test Parquet
        parquet_path = tmp_path / "data.parquet"
        df.write_parquet(parquet_path)
        session = ResearchSession.load(parquet_path)
        assert session is not None


class TestFactoriumExports:
    """Test that ResearchSession is exported from main module."""

    def test_research_session_in_main_exports(self):
        """ResearchSession should be importable from factorium."""
        from factorium import ResearchSession

        assert ResearchSession is not None


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
