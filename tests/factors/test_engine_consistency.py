import pytest
import numpy as np
import pandas as pd
import polars as pl

from ._legacy_engine import PolarsEngine, PandasEngine


def _to_pandas_df(result):
    if isinstance(result, pl.DataFrame):
        return result.to_pandas()
    return result


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "start_time": [1, 2, 3, 4, 5] * 2,
            "end_time": [2, 3, 4, 5, 6] * 2,
            "symbol": ["A"] * 5 + ["B"] * 5,
            "factor": np.random.randn(10),
        }
    )


@pytest.fixture
def cs_sample_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "start_time": [1, 1, 1, 2, 2, 2],
            "end_time": [2, 2, 2, 3, 3, 3],
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


@pytest.fixture(params=[PolarsEngine(), PandasEngine()])
def engine(request):
    return request.param


class TestEngineConsistency:
    def test_ts_sum(self, engine, sample_df):
        result = engine.ts_sum(sample_df, window=3)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(sample_df)
        assert result["factor"][:2].isna().all()

    def test_ts_mean(self, engine, sample_df):
        result = engine.ts_mean(sample_df, window=3)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(sample_df)
        assert result["factor"][:2].isna().all()

    def test_ts_std(self, engine, sample_df):
        result = engine.ts_std(sample_df, window=3)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(sample_df)
        assert result["factor"][:2].isna().all()

    def test_ts_min(self, engine, sample_df):
        result = engine.ts_min(sample_df, window=3)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(sample_df)
        assert result["factor"][:2].isna().all()

    def test_ts_max(self, engine, sample_df):
        result = engine.ts_max(sample_df, window=3)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(sample_df)
        assert result["factor"][:2].isna().all()

    def test_ts_shift(self, engine, sample_df):
        result = engine.ts_shift(sample_df, period=2)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(sample_df)
        assert result["factor"][:2].isna().all()

    def test_ts_diff(self, engine, sample_df):
        result = engine.ts_diff(sample_df, period=1)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(sample_df)

    def test_cs_rank(self, engine, cs_sample_df):
        result = engine.cs_rank(cs_sample_df)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(cs_sample_df)

        valid = result["factor"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_cs_zscore(self, engine, cs_sample_df):
        result = engine.cs_zscore(cs_sample_df)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(cs_sample_df)

        for end_time in result["end_time"].unique():
            time_data = result[result["end_time"] == end_time]
            mean = time_data["factor"].mean()
            assert abs(mean) < 1e-10

    def test_cs_demean(self, engine, cs_sample_df):
        result = engine.cs_demean(cs_sample_df)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(cs_sample_df)

        for end_time in result["end_time"].unique():
            time_data = result[result["end_time"] == end_time]
            mean = time_data["factor"].mean()
            assert abs(mean) < 1e-10

    def test_ts_skewness(self, engine, sample_df):
        result = engine.ts_skewness(sample_df, window=3)
        if isinstance(engine, PolarsEngine):
            assert isinstance(result, pl.DataFrame)
        else:
            assert isinstance(result, pd.DataFrame)
        result = _to_pandas_df(result)
        assert len(result) == len(sample_df)
        # First 2 rows should be NaN (window=3, min_samples=3)
        assert result["factor"][:2].isna().all()


class TestEngineNumericalConsistency:
    def test_ts_sum_values_match(self, sample_df):
        polars_engine = PolarsEngine()
        pandas_engine = PandasEngine()

        result_polars = _to_pandas_df(polars_engine.ts_sum(sample_df.copy(), window=3))
        result_pandas = pandas_engine.ts_sum(sample_df.copy(), window=3)

        valid_mask = result_polars["factor"].notna() & result_pandas["factor"].notna()
        np.testing.assert_allclose(
            result_polars.loc[valid_mask, "factor"].values,
            result_pandas.loc[valid_mask, "factor"].values,
            rtol=1e-10,
        )

    def test_ts_mean_values_match(self, sample_df):
        polars_engine = PolarsEngine()
        pandas_engine = PandasEngine()

        result_polars = _to_pandas_df(polars_engine.ts_mean(sample_df.copy(), window=3))
        result_pandas = pandas_engine.ts_mean(sample_df.copy(), window=3)

        valid_mask = result_polars["factor"].notna() & result_pandas["factor"].notna()
        np.testing.assert_allclose(
            result_polars.loc[valid_mask, "factor"].values,
            result_pandas.loc[valid_mask, "factor"].values,
            rtol=1e-10,
        )

    def test_cs_rank_values_match(self, cs_sample_df):
        polars_engine = PolarsEngine()
        pandas_engine = PandasEngine()

        result_polars = _to_pandas_df(polars_engine.cs_rank(cs_sample_df.copy()))
        result_pandas = pandas_engine.cs_rank(cs_sample_df.copy())

        valid_mask = result_polars["factor"].notna() & result_pandas["factor"].notna()
        np.testing.assert_allclose(
            result_polars.loc[valid_mask, "factor"].values,
            result_pandas.loc[valid_mask, "factor"].values,
            rtol=1e-9,
            atol=1e-12,
        )

    def test_ts_skewness_values_match(self, sample_df):
        polars_engine = PolarsEngine()
        pandas_engine = PandasEngine()

        result_polars = _to_pandas_df(polars_engine.ts_skewness(sample_df.copy(), window=3))
        result_pandas = pandas_engine.ts_skewness(sample_df.copy(), window=3)

        valid_mask = result_polars["factor"].notna() & result_pandas["factor"].notna()
        np.testing.assert_allclose(
            result_polars.loc[valid_mask, "factor"].values,
            result_pandas.loc[valid_mask, "factor"].values,
            rtol=1e-9,
            atol=1e-12,
        )
