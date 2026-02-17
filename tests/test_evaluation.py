import pandas as pd
import numpy as np
import pytest
from factorium.factors.core import Factor

def test_factor_evaluation_flow():
    # Create dummy price data (upward trend)
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOG']
    
    price_data = []
    for d in dates:
        for s in symbols:
            # Price increases over time
            p = 100 + (d - dates[0]).days * 2 + np.random.randn()
            price_data.append([d, d, s, p])
            
    prices_df = pd.DataFrame(price_data, columns=['start_time', 'end_time', 'symbol', 'factor'])
    prices_factor = Factor(prices_df, name='close')
    
    # Create dummy signal factor 
    signal_data = []
    for i, d in enumerate(dates):
        for s in symbols:
            sig = np.random.randn()
            signal_data.append([d, d, s, sig])
            
    signal_df = pd.DataFrame(signal_data, columns=['start_time', 'end_time', 'symbol', 'factor'])
    signal_factor = Factor(signal_df, name='signal')
    
    # Run eval method with plot
    import os
    plot_path = "test_eval_plot.png"
    results = signal_factor.eval(prices_factor, periods=[1, 2], quantiles=2, save_path=plot_path)
    
    # Assertions
    assert 'ic_mean' in results
    assert 'ic_ir' in results
    assert 'layer_returns' in results
    assert 'turnover_mean' in results
    assert isinstance(results['ic_mean'], pd.Series)
    assert 1 in results['layer_returns']
    assert 2 in results['layer_returns']
    
    # Check if plot was saved
    assert os.path.exists(plot_path)
    if os.path.exists(plot_path):
        os.remove(plot_path) # Cleanup
