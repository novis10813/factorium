"""Tests for bar data structures."""

import pytest
import pandas as pd
import numpy as np
from factorium import TimeBar, TickBar, VolumeBar, DollarBar, AggBar


@pytest.fixture
def sample_trades():
    """Create sample trade data for testing."""
    np.random.seed(42)
    n_trades = 1000
    
    # Generate timestamps over 1 hour
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
    timestamps = base_ts + np.arange(n_trades) * 3600  # ~3.6 seconds apart
    
    # Generate prices with random walk
    prices = 100 + np.cumsum(np.random.randn(n_trades) * 0.1)
    
    # Generate volumes
    volumes = np.abs(np.random.randn(n_trades)) * 10 + 1
    
    df = pd.DataFrame({
        'ts_init': timestamps,
        'price': prices,
        'size': volumes,
        'symbol': 'BTCUSDT',
        'is_buyer_maker': np.random.choice([True, False], n_trades)
    })
    
    return df


def test_timebar_creation(sample_trades):
    """Test TimeBar creation."""
    bar = TimeBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_ms=60_000  # 1 minute
    )
    
    assert len(bar) > 0
    assert 'open' in bar.bars.columns
    assert 'high' in bar.bars.columns
    assert 'low' in bar.bars.columns
    assert 'close' in bar.bars.columns
    assert 'volume' in bar.bars.columns
    assert 'vwap' in bar.bars.columns
    assert 'start_time' in bar.bars.columns
    assert 'end_time' in bar.bars.columns
    assert 'symbol' in bar.bars.columns


def test_tickbar_creation(sample_trades):
    """Test TickBar creation."""
    bar = TickBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_ticks=100
    )
    
    assert len(bar) > 0
    expected_bars = len(sample_trades) // 100
    assert len(bar) >= expected_bars


def test_volumebar_creation(sample_trades):
    """Test VolumeBar creation."""
    # Calculate total volume to set appropriate threshold
    total_volume = sample_trades['size'].sum()
    target_volume = total_volume / 10  # Target ~10 bars
    
    bar = VolumeBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_volume=target_volume
    )
    
    assert len(bar) > 0


def test_dollarbar_creation(sample_trades):
    """Test DollarBar creation."""
    # Calculate total dollar volume
    total_dollar = (sample_trades['size'] * sample_trades['price']).sum()
    target_dollar = total_dollar / 10  # Target ~10 bars
    
    bar = DollarBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_dollar=int(target_dollar)
    )
    
    assert len(bar) > 0


def test_bar_apply_transformation(sample_trades):
    """Test applying transformations to bars."""
    bar = TimeBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_ms=60_000
    )
    
    bar.apply({
        'returns': lambda bars: bars['close'].pct_change(),
        'sma_5': lambda bars: bars['close'].rolling(5).mean(),
    })
    
    assert 'returns' in bar.bars.columns
    assert 'sma_5' in bar.bars.columns


def test_aggbar_from_bars(sample_trades):
    """Test AggBar creation from multiple bars."""
    # Create two bars with different symbols
    df1 = sample_trades.copy()
    df1['symbol'] = 'BTCUSDT'
    
    df2 = sample_trades.copy()
    df2['symbol'] = 'ETHUSDT'
    df2['price'] = df2['price'] / 10  # Different price scale
    
    bar1 = TimeBar(df1, timestamp_col='ts_init', price_col='price', volume_col='size', interval_ms=60_000)
    bar2 = TimeBar(df2, timestamp_col='ts_init', price_col='price', volume_col='size', interval_ms=60_000)
    
    agg = AggBar([bar1, bar2])
    
    assert len(agg.symbols) == 2
    assert 'BTCUSDT' in agg.symbols
    assert 'ETHUSDT' in agg.symbols


def test_aggbar_getitem_factor(sample_trades):
    """Test extracting a Factor from AggBar."""
    bar = TimeBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_ms=60_000
    )
    
    agg = AggBar([bar])
    close_factor = agg['close']
    
    assert close_factor.name == 'close'
    assert len(close_factor) == len(bar)


def test_aggbar_slice(sample_trades):
    """Test slicing AggBar by time."""
    bar = TimeBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_ms=60_000
    )
    
    agg = AggBar([bar])
    
    # Get timestamps
    start_ts = agg.data['start_time'].min()
    mid_ts = start_ts + (agg.data['end_time'].max() - start_ts) // 2
    
    sliced = agg.slice(start=start_ts, end=mid_ts)
    
    assert len(sliced) < len(agg)
    assert sliced.data['start_time'].min() >= start_ts
    assert sliced.data['end_time'].max() <= mid_ts


def test_timebar_has_vwap(sample_trades):
    """Test that TimeBar includes VWAP column."""
    bar = TimeBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_ms=60_000
    )
    
    assert 'vwap' in bar.bars.columns
    assert not bar.bars['vwap'].isna().any()
    assert (bar.bars['vwap'] > 0).all()  # VWAP should be positive


def test_vwap_calculation(sample_trades):
    """Test that VWAP is calculated correctly."""
    bar = TimeBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_ms=60_000
    )
    
    # Verify VWAP calculation: sum(price * volume) / sum(volume)
    for idx, row in bar.bars.iterrows():
        # Get original trades for this bar
        start_time = row['start_time']
        end_time = row['end_time']
        
        bar_trades = sample_trades[
            (sample_trades['ts_init'] >= start_time) & 
            (sample_trades['ts_init'] < end_time)
        ]
        
        if len(bar_trades) > 0:
            expected_vwap = (bar_trades['price'] * bar_trades['size']).sum() / bar_trades['size'].sum()
            assert abs(row['vwap'] - expected_vwap) < 1e-10, \
                f"VWAP mismatch at bar {idx}: expected {expected_vwap}, got {row['vwap']}"


def test_all_bar_types_have_vwap(sample_trades):
    """Test that all bar types include VWAP column."""
    # Test TickBar
    tick_bar = TickBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_ticks=100
    )
    assert 'vwap' in tick_bar.bars.columns
    assert not tick_bar.bars['vwap'].isna().any()
    
    # Test VolumeBar
    total_volume = sample_trades['size'].sum()
    target_volume = total_volume / 10
    volume_bar = VolumeBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_volume=target_volume
    )
    assert 'vwap' in volume_bar.bars.columns
    assert not volume_bar.bars['vwap'].isna().any()
    
    # Test DollarBar
    total_dollar = (sample_trades['size'] * sample_trades['price']).sum()
    target_dollar = total_dollar / 10
    dollar_bar = DollarBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_dollar=int(target_dollar)
    )
    assert 'vwap' in dollar_bar.bars.columns
    assert not dollar_bar.bars['vwap'].isna().any()


def test_vwap_between_price_range(sample_trades):
    """Test that VWAP is between low and high prices for each bar."""
    bar = TimeBar(
        sample_trades,
        timestamp_col='ts_init',
        price_col='price',
        volume_col='size',
        interval_ms=60_000
    )
    
    for _, row in bar.bars.iterrows():
        assert row['low'] <= row['vwap'] <= row['high'], \
            f"VWAP {row['vwap']} should be between low {row['low']} and high {row['high']}"
