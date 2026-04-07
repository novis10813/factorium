import polars as pl

from factorium.backtest.normalizers import (
    MinMaxNormalizer,
    RankNormalizer,
    RawNormalizer,
    ZScoreNormalizer,
)


class TestRawNormalizer:
    def test_passthrough(self):
        """RawNormalizer should not modify the signal."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [0.8, -0.5, 2.0],
        })
        result = RawNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"].to_list() == [0.8, -0.5, 2.0]

    def test_null_passthrough(self):
        """RawNormalizer should preserve nulls."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [0.8, None, 2.0],
        })
        result = RawNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"].to_list() == [0.8, None, 2.0]


class TestRankNormalizer:
    def test_output_range_zero_to_one(self):
        """RankNormalizer output should be in [0, 1]."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [10.0, 30.0, 20.0, 40.0],
        })
        result = RankNormalizer().normalize(df, "signal", "end_time")
        values = result["signal"]
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_preserves_ranking_order(self):
        """Higher signal should get higher rank value."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, 3.0, 2.0],
        })
        result = RankNormalizer().normalize(df, "signal", "end_time")
        vals = result["signal"].to_list()
        # B (3.0) > C (2.0) > A (1.0), so rank(B) > rank(C) > rank(A)
        assert vals[1] > vals[2] > vals[0]

    def test_cross_sectional_per_group(self):
        """Ranking should be independent per group."""
        df = pl.DataFrame({
            "end_time": [1000, 1000, 2000, 2000],
            "symbol": ["A", "B", "A", "B"],
            "signal": [10.0, 20.0, 50.0, 5.0],
        })
        result = RankNormalizer().normalize(df, "signal", "end_time")
        # Group 1000: B > A, Group 2000: A > B
        g1 = result.filter(pl.col("end_time") == 1000)["signal"].to_list()
        g2 = result.filter(pl.col("end_time") == 2000)["signal"].to_list()
        assert g1[1] > g1[0]  # B > A in group 1000
        assert g2[0] > g2[1]  # A > B in group 2000

    def test_null_signal_stays_null(self):
        """Null signals should remain null after ranking."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, None, 3.0],
        })
        result = RankNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"][1] is None


class TestZScoreNormalizer:
    def test_mean_near_zero(self):
        """Cross-sectional z-score mean should be ~0."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = ZScoreNormalizer().normalize(df, "signal", "end_time")
        assert abs(result["signal"].mean()) < 1e-10

    def test_std_near_one(self):
        """Cross-sectional z-score std should be ~1."""
        df = pl.DataFrame({
            "end_time": [1000] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "signal": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = ZScoreNormalizer().normalize(df, "signal", "end_time")
        assert abs(result["signal"].std() - 1.0) < 0.1

    def test_zero_std_produces_null(self):
        """When all signals are identical (std=0), output should be null."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [5.0, 5.0, 5.0],
        })
        result = ZScoreNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"].null_count() == 3

    def test_null_signal_stays_null(self):
        """Null signals should remain null after z-score."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [1.0, None, 3.0],
        })
        result = ZScoreNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"][1] is None


class TestMinMaxNormalizer:
    def test_output_range_zero_to_one(self):
        """MinMaxNormalizer output should be in [0, 1]."""
        df = pl.DataFrame({
            "end_time": [1000] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [10.0, 30.0, 20.0, 40.0],
        })
        result = MinMaxNormalizer().normalize(df, "signal", "end_time")
        values = result["signal"]
        assert values.min() >= -1e-10
        assert values.max() <= 1.0 + 1e-10

    def test_min_maps_to_zero_max_maps_to_one(self):
        """Min value should map to 0, max to 1."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [10.0, 20.0, 30.0],
        })
        result = MinMaxNormalizer().normalize(df, "signal", "end_time")
        vals = result["signal"].to_list()
        assert abs(vals[0] - 0.0) < 1e-10  # min -> 0
        assert abs(vals[2] - 1.0) < 1e-10  # max -> 1

    def test_zero_range_produces_null(self):
        """When all signals are identical (range=0), output should be null."""
        df = pl.DataFrame({
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [5.0, 5.0, 5.0],
        })
        result = MinMaxNormalizer().normalize(df, "signal", "end_time")
        assert result["signal"].null_count() == 3

    def test_cross_sectional_per_group(self):
        """MinMax should be independent per group."""
        df = pl.DataFrame({
            "end_time": [1000, 1000, 2000, 2000],
            "symbol": ["A", "B", "A", "B"],
            "signal": [10.0, 20.0, 100.0, 200.0],
        })
        result = MinMaxNormalizer().normalize(df, "signal", "end_time")
        g1 = result.filter(pl.col("end_time") == 1000)["signal"].to_list()
        g2 = result.filter(pl.col("end_time") == 2000)["signal"].to_list()
        # Both groups should have min=0, max=1
        assert abs(g1[0] - 0.0) < 1e-10
        assert abs(g1[1] - 1.0) < 1e-10
        assert abs(g2[0] - 0.0) < 1e-10
        assert abs(g2[1] - 1.0) < 1e-10
