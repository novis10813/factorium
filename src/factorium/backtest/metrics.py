"""Performance metrics calculation for backtesting."""

from typing import Dict
import numpy as np
import pandas as pd

from .utils import MAX_PERIODS_PER_YEAR, MIN_PERIODS_PER_YEAR, safe_divide


def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 365.0 * 24,
) -> Dict[str, float]:
    if not MIN_PERIODS_PER_YEAR <= periods_per_year <= MAX_PERIODS_PER_YEAR:
        raise ValueError(
            f"periods_per_year must be between {MIN_PERIODS_PER_YEAR} and {MAX_PERIODS_PER_YEAR}, "
            f"got {periods_per_year}"
        )

    nan_result = {
        "total_return": np.nan,
        "annual_return": np.nan,
        "annual_volatility": np.nan,
        "sharpe_ratio": np.nan,
        "sortino_ratio": np.nan,
        "calmar_ratio": np.nan,
        "max_drawdown": np.nan,
        "var_95": np.nan,
        "cvar_95": np.nan,
        "win_rate": np.nan,
        "profit_factor": np.nan,
    }

    if returns.empty or returns.isna().all():
        return nan_result

    returns = returns.dropna()
    if len(returns) < 2:
        return nan_result

    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    annual_volatility = float(returns.std() * np.sqrt(periods_per_year))

    excess_return = annual_return - risk_free_rate
    sharpe_ratio = safe_divide(excess_return, annual_volatility, default=0.0)

    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = float(downside_returns.std() * np.sqrt(periods_per_year))
        sortino_ratio = safe_divide(excess_return, downside_std, default=0.0)
    else:
        sortino_ratio = np.inf if excess_return > 0 else 0.0

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    calmar_ratio = safe_divide(annual_return, abs(max_drawdown), default=0.0)

    # VaR & CVaR (95%)
    var_95 = float(returns.quantile(0.05))
    cvar_mask = returns <= var_95
    cvar_95 = float(returns[cvar_mask].mean()) if cvar_mask.any() else var_95

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = safe_divide(float(len(wins)), float(len(returns)), default=0.0)

    total_losses = abs(losses.sum())
    profit_factor = (
        safe_divide(float(wins.sum()), total_losses, default=np.inf)
        if len(losses) > 0
        else (np.inf if wins.sum() > 0 else 0.0)
    )

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": annual_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "calmar_ratio": float(calmar_ratio),
        "max_drawdown": max_drawdown,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "win_rate": float(win_rate),
        "profit_factor": profit_factor,
    }
