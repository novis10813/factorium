"""Tests for Factor plotting functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from factorium import Factor, AggBar


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture(scope="module")
def sample_aggbar_for_plotting():
    """
    Creates a synthetic AggBar with multiple symbols and more data points
    for plotting visualization tests.
    """
    # Generate more time points for better visualization
    dates = pd.date_range(start="2025-01-01", periods=50, freq="1h")
    timestamps = (dates.astype(np.int64) // 10**6).astype(int)
    
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    data_list = []
    
    np.random.seed(42)  # For reproducible results
    
    for symbol in symbols:
        # Generate different price patterns for each symbol
        if symbol == "BTCUSDT":
            base_price = 100
            volatility = 2.0
            trend = np.linspace(0, 20, 50)
        elif symbol == "ETHUSDT":
            base_price = 50
            volatility = 1.5
            trend = np.linspace(0, 15, 50)
        elif symbol == "BNBUSDT":
            base_price = 20
            volatility = 1.0
            trend = np.linspace(0, 10, 50)
        else:  # SOLUSDT
            base_price = 30
            volatility = 1.2
            trend = np.linspace(0, 12, 50)
        
        # Generate price series with trend and noise
        noise = np.random.randn(50) * volatility
        prices = base_price + trend + np.cumsum(noise)
        
        df = pd.DataFrame({
            "start_time": timestamps,
            "end_time": timestamps + 3600000,  # +1 hour
            "symbol": symbol,
            "close": prices,
            "open": prices * 0.98,  # Slightly lower open prices
            "high": prices * 1.02,
            "low": prices * 0.96,
            "volume": np.abs(np.random.randn(50) * 1000) + 1000
        })
        data_list.append(df)
    
    df_combined = pd.concat(data_list, ignore_index=True)
    return AggBar(df_combined)


@pytest.fixture
def factor_close(sample_aggbar_for_plotting):
    """Factor for close prices."""
    return sample_aggbar_for_plotting["close"]


@pytest.fixture
def output_dir(tmp_path):
    """Create output directory for saved plots."""
    output = tmp_path / "plots"
    output.mkdir()
    return output


# ==========================================
# Test Cases
# ==========================================

def test_plot_timeseries(factor_close, output_dir):
    """Test time series plot."""
    fig = factor_close.plot(plot_type='timeseries', figsize=(14, 6))
    assert fig is not None
    
    output_path = output_dir / "timeseries.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    assert output_path.exists()
    print(f"\n✓ Saved timeseries plot to: {output_path}")


def test_plot_timeseries_with_symbols(factor_close, output_dir):
    """Test time series plot with symbol filtering."""
    fig = factor_close.plot(
        plot_type='timeseries',
        symbols=['BTCUSDT', 'ETHUSDT'],
        figsize=(12, 6)
    )
    assert fig is not None
    
    output_path = output_dir / "timeseries_filtered.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    assert output_path.exists()
    print(f"\n✓ Saved filtered timeseries plot to: {output_path}")


def test_plot_heatmap(factor_close, output_dir):
    """Test heatmap plot."""
    fig = factor_close.plot(plot_type='heatmap', figsize=(14, 8))
    assert fig is not None
    
    output_path = output_dir / "heatmap.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    assert output_path.exists()
    print(f"\n✓ Saved heatmap plot to: {output_path}")


def test_plot_distribution_histogram(factor_close, output_dir):
    """Test distribution plot (histogram)."""
    fig = factor_close.plot(
        plot_type='distribution',
        dist_type='histogram',
        figsize=(12, 6)
    )
    assert fig is not None
    
    output_path = output_dir / "distribution_histogram.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    assert output_path.exists()
    print(f"\n✓ Saved distribution histogram to: {output_path}")


def test_plot_distribution_density(factor_close, output_dir):
    """Test distribution plot (density)."""
    fig = factor_close.plot(
        plot_type='distribution',
        dist_type='density',
        figsize=(12, 6)
    )
    assert fig is not None
    
    output_path = output_dir / "distribution_density.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    assert output_path.exists()
    print(f"\n✓ Saved distribution density to: {output_path}")


def test_plot_with_time_filter(factor_close, output_dir):
    """Test plot with time range filtering."""
    from datetime import datetime
    
    fig = factor_close.plot(
        plot_type='timeseries',
        start_time=datetime(2025, 1, 1, 10),
        end_time=datetime(2025, 1, 1, 20),
        figsize=(12, 6)
    )
    assert fig is not None
    
    output_path = output_dir / "timeseries_time_filtered.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    assert output_path.exists()
    print(f"\n✓ Saved time-filtered plot to: {output_path}")


def test_plot_all_types(factor_close, output_dir):
    """Generate all plot types and save them."""
    plot_types = [
        ('timeseries', {}),
        ('heatmap', {'figsize': (14, 8)}),
        ('distribution', {'dist_type': 'histogram'}),
        ('distribution', {'dist_type': 'density'}),
    ]
    
    saved_files = []
    for plot_type, kwargs in plot_types:
        fig = factor_close.plot(plot_type=plot_type, **kwargs)
        assert fig is not None
        
        filename = f"all_types_{plot_type}.png"
        if plot_type == 'distribution':
            filename = f"all_types_{plot_type}_{kwargs.get('dist_type', 'histogram')}.png"
        
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        assert output_path.exists()
        saved_files.append(output_path)
        print(f"\n✓ Saved {plot_type} plot to: {output_path}")
    
    print(f"\n✓ All {len(saved_files)} plot types saved successfully!")

