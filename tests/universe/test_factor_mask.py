import polars as pl

from factorium import AggBar
from factorium.factors.analyzer import FactorAnalyzer


def _make_aggbar() -> AggBar:
    rows = []
    base_ts = 1_700_000_000_000
    for i in range(8):
        ts = base_ts + i * 3_600_000
        rows.extend(
            [
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "BTCUSDT",
                    "open": 100 + i,
                    "high": 101 + i,
                    "low": 99 + i,
                    "close": 100 + i,
                    "volume": 1_000.0,
                    "alpha": float(i + 1),
                    "in_checklist": True,
                },
                {
                    "start_time": ts,
                    "end_time": ts + 3_600_000,
                    "symbol": "DOGEUSDT",
                    "open": 10 + i,
                    "high": 11 + i,
                    "low": 9 + i,
                    "close": 10 + i,
                    "volume": 1_000.0,
                    "alpha": float(100 - i),
                    "in_checklist": False,
                },
            ]
        )
    return AggBar(pl.DataFrame(rows))


def test_factor_eval_mask_is_backward_compatible_when_none() -> None:
    agg = _make_aggbar()
    factor = agg["alpha"]

    with_mask_none = factor.eval(prices=agg, periods=1, quantiles=2, mask=None)
    without_mask = factor.eval(prices=agg, periods=1, quantiles=2)

    assert with_mask_none.ic_series.equals(without_mask.ic_series)


def test_factor_analyzer_applies_mask_from_aggbar() -> None:
    agg = _make_aggbar()
    factor = agg["alpha"]

    analyzer = FactorAnalyzer(factor=factor, prices=agg, quantiles=2, mask="in_checklist")
    prepared = analyzer.prepare_data(periods=[1], price_col="close")

    assert set(prepared["symbol"].to_list()) == {"BTCUSDT"}


def test_factor_eval_with_mask_changes_universe_used_for_analysis() -> None:
    agg = _make_aggbar()
    factor = agg["alpha"]

    unmasked = factor.eval(prices=agg, periods=1, quantiles=2)
    masked = factor.eval(prices=agg, periods=1, quantiles=2, mask="in_checklist")

    assert len(masked.ic_series) <= len(unmasked.ic_series)
