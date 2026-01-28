import polars as pl
from factorium.backtest.utils import neutralize_weights_polars


def test_neutralize_weights_polars():
    """Should create market neutral weights."""
    df = pl.DataFrame(
        {
            "end_time": [1000] * 3,
            "symbol": ["A", "B", "C"],
            "signal": [0.8, 0.5, 0.2],
        }
    )

    result = neutralize_weights_polars(df, "signal", "end_time")

    # Weights should sum to zero
    assert abs(result["weight"].sum()) < 1e-10
    # Absolute weights should sum to 1
    assert abs(result["weight"].abs().sum() - 1.0) < 1e-10
