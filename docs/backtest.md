# Backtesting Engine 設計文檔

> 本文檔定義 Factorium 回測引擎的設計規格與實作細節。

## 1. 概述

### 1.1 目標

為 Factorium 提供一個**簡潔、高效、可擴展**的回測引擎，支援：

- 基於因子信號的策略回測
- 多標的同時交易
- 交易成本計算
- 美元中性（Dollar-neutral）策略
- 完整的績效指標計算

### 1.2 設計原則

| 原則 | 說明 |
|------|------|
| **與現有架構一致** | 使用 `Factor`、`AggBar` 的資料格式（`start_time`, `end_time`, `symbol`, `factor`） |
| **避免前視偏差** | 使用 T-1 時刻的信號在 T 時刻交易 |
| **NaN 安全** | 遵循 `safe_` 函數模式，遇到 NaN 時跳過或傳遞 |
| **可組合** | 與 `FactorEvaluator` 互補，可串接使用 |

---

## 2. 模組結構

```
src/factorium/backtest/
├── __init__.py          # 公開 API: Backtester, BacktestResult, calculate_metrics
├── backtester.py        # Backtester 主類別
├── portfolio.py         # Portfolio 持倉管理
├── metrics.py           # 績效指標計算
└── utils.py             # 輔助函數（持倉計算、中性化等）
```

---

## 3. 資料格式

### 3.1 輸入格式

| 資料 | 類型 | 格式 |
|------|------|------|
| `prices` | `AggBar` | OHLCV 多標的資料，含 `start_time`, `end_time`, `symbol`, `open`, `high`, `low`, `close`, `volume` |
| `signal` | `Factor` | 策略信號因子，含 `start_time`, `end_time`, `symbol`, `factor` |

### 3.2 時間戳格式

- 所有時間戳使用 **int64 毫秒**（與現有 `Factor`/`AggBar` 一致）
- 按 `end_time` 對齊（bar 結束時間點）

---

## 4. 類別設計

### 4.1 Portfolio（持倉管理）

```python
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

@dataclass
class Portfolio:
    """
    管理持倉、現金、交易記錄。
    
    Attributes:
        initial_capital: 初始資金
        cash: 當前現金
        positions: 當前持倉 {symbol: quantity}
        history: 淨值歷史記錄
        trade_log: 交易記錄
    """
    initial_capital: float = 10000.0
    cash: float = field(init=False)
    positions: Dict[str, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    trade_log: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.cash = self.initial_capital
    
    def execute_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        cost_rates: tuple[float, float],
        timestamp: int
    ) -> None:
        """
        執行交易。
        
        Args:
            symbol: 標的代碼
            quantity: 交易數量（正數買入，負數賣出）
            price: 成交價格
            cost_rates: (買入費率, 賣出費率)
            timestamp: 時間戳（毫秒）
        """
        if quantity == 0:
            return
            
        trade_value = abs(quantity * price)
        cost_rate = cost_rates[0] if quantity > 0 else cost_rates[1]
        cost = trade_value * cost_rate
        
        # 更新現金
        if quantity > 0:  # 買入
            self.cash -= (trade_value + cost)
        else:  # 賣出
            self.cash += (trade_value - cost)
        
        # 更新持倉
        current_qty = self.positions.get(symbol, 0.0)
        new_qty = current_qty + quantity
        
        if abs(new_qty) < 1e-10:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty
        
        # 記錄交易
        self.trade_log.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'trade_value': trade_value,
        })
    
    def get_market_value(self, prices: pd.Series) -> float:
        """
        計算持倉市值。
        
        Args:
            prices: {symbol: price} 的 Series
            
        Returns:
            持倉總市值
        """
        market_value = 0.0
        for symbol, qty in self.positions.items():
            if symbol in prices.index:
                market_value += qty * prices[symbol]
        return market_value
    
    def get_total_value(self, prices: pd.Series) -> float:
        """計算總資產（現金 + 持倉市值）"""
        return self.cash + self.get_market_value(prices)
    
    def record_snapshot(self, timestamp: int, prices: pd.Series) -> None:
        """
        記錄當前時刻的資產快照。
        
        Args:
            timestamp: 時間戳（毫秒）
            prices: 當前價格
        """
        self.history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'market_value': self.get_market_value(prices),
            'total_value': self.get_total_value(prices),
            'num_positions': len(self.positions),
        })
    
    def get_history_df(self) -> pd.DataFrame:
        """將歷史記錄轉為 DataFrame"""
        return pd.DataFrame(self.history)
    
    def get_trade_log_df(self) -> pd.DataFrame:
        """將交易記錄轉為 DataFrame"""
        return pd.DataFrame(self.trade_log)
```

---

### 4.2 Backtester（回測引擎）

```python
from typing import Literal, Union, Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

from ..factors.core import Factor
from ..aggbar import AggBar
from .portfolio import Portfolio
from .metrics import calculate_metrics
from .utils import neutralize_weights


@dataclass
class BacktestResult:
    """
    回測結果容器。
    
    Attributes:
        equity_curve: 淨值曲線
        returns: 報酬率序列
        metrics: 績效指標字典
        trades: 交易記錄 DataFrame
        portfolio_history: 投資組合歷史 DataFrame
    """
    equity_curve: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]
    trades: pd.DataFrame
    portfolio_history: pd.DataFrame


class Backtester:
    """
    因子策略回測引擎。
    
    使用因子信號生成持倉權重，模擬交易並計算績效。
    
    Args:
        prices: OHLCV 資料（AggBar）
        signal: 策略因子信號（Factor）
        entry_price: 入場價格欄位名稱，預設 "close"
        transaction_cost: 交易成本，可為單一數值或 (買入費率, 賣出費率)
        initial_capital: 初始資金
        full_rebalance: 是否完全調倉（先平倉再建倉）
        neutralization: 中性化方式 ("market" 或 "none")
    
    Example:
        >>> from factorium import AggBar
        >>> from factorium.backtest import Backtester
        >>> 
        >>> agg = AggBar.from_parquet("data.parquet")
        >>> close = agg["close"]
        >>> signal = (close / close.ts_shift(20) - 1).cs_rank()
        >>> 
        >>> bt = Backtester(
        ...     prices=agg,
        ...     signal=signal,
        ...     transaction_cost=(0.0003, 0.0003),
        ...     neutralization="market",
        ... )
        >>> result = bt.run()
        >>> print(result.metrics)
    """
    
    def __init__(
        self,
        prices: AggBar,
        signal: Factor,
        entry_price: str = "close",
        transaction_cost: Union[float, tuple[float, float]] = 0.0003,
        initial_capital: float = 10000.0,
        full_rebalance: bool = False,
        neutralization: Literal["market", "none"] = "market",
    ):
        self.prices = prices
        self.signal = signal
        self.entry_price = entry_price
        self.initial_capital = initial_capital
        self.full_rebalance = full_rebalance
        self.neutralization = neutralization
        
        # 處理交易成本
        if isinstance(transaction_cost, (int, float)):
            self.cost_rates = (float(transaction_cost), float(transaction_cost))
        else:
            self.cost_rates = transaction_cost
        
        # 驗證資料
        self._validate_inputs()
        
        # 內部狀態
        self._portfolio: Optional[Portfolio] = None
        self._result: Optional[BacktestResult] = None
    
    def _validate_inputs(self) -> None:
        """驗證輸入資料"""
        if self.entry_price not in self.prices.cols:
            raise ValueError(f"entry_price '{self.entry_price}' not found in prices")
        
        # 檢查 signal 和 prices 是否有共同時間點
        signal_times = set(self.signal.data['end_time'].unique())
        price_times = set(self.prices.data['end_time'].unique())
        common_times = signal_times & price_times
        
        if len(common_times) < 2:
            raise ValueError("signal and prices must have at least 2 common timestamps")
    
    def _get_common_timestamps(self) -> np.ndarray:
        """取得 signal 和 prices 共同的時間戳（已排序）"""
        signal_times = set(self.signal.data['end_time'].unique())
        price_times = set(self.prices.data['end_time'].unique())
        common = sorted(signal_times & price_times)
        return np.array(common)
    
    def _get_prices_at(self, timestamp: int) -> pd.Series:
        """取得特定時間點的價格"""
        df = self.prices.data
        mask = df['end_time'] == timestamp
        prices = df.loc[mask].set_index('symbol')[self.entry_price]
        return prices
    
    def _get_signal_at(self, timestamp: int) -> pd.Series:
        """取得特定時間點的信號"""
        df = self.signal.data
        mask = df['end_time'] == timestamp
        signals = df.loc[mask].set_index('symbol')['factor']
        return signals
    
    def _calculate_target_weights(self, signals: pd.Series) -> pd.Series:
        """
        根據信號計算目標權重。
        
        Args:
            signals: 原始信號值
            
        Returns:
            目標權重（和為 0 如果是 market neutral）
        """
        # 移除 NaN
        signals = signals.dropna()
        
        if len(signals) == 0:
            return pd.Series(dtype=float)
        
        if self.neutralization == "none":
            # 簡單歸一化
            total = signals.abs().sum()
            if total == 0:
                return pd.Series(0.0, index=signals.index)
            return signals / total
        
        elif self.neutralization == "market":
            # 美元中性：多空平衡
            return neutralize_weights(signals)
        
        else:
            raise ValueError(f"Unknown neutralization: {self.neutralization}")
    
    def _calculate_target_holdings(
        self,
        weights: pd.Series,
        prices: pd.Series,
        total_value: float
    ) -> pd.Series:
        """
        根據權重計算目標持倉數量。
        
        Args:
            weights: 目標權重
            prices: 當前價格
            total_value: 投資組合總價值
            
        Returns:
            目標持倉數量
        """
        # 對齊 symbols
        common_symbols = weights.index.intersection(prices.index)
        weights = weights.loc[common_symbols]
        prices = prices.loc[common_symbols]
        
        # 計算目標市值
        target_values = weights * total_value
        
        # 計算目標數量
        target_quantities = target_values / prices
        
        return target_quantities
    
    def _generate_orders(
        self,
        target_holdings: pd.Series,
        current_holdings: Dict[str, float]
    ) -> Dict[str, float]:
        """
        生成訂單（目標持倉 - 當前持倉）。
        
        Args:
            target_holdings: 目標持倉
            current_holdings: 當前持倉
            
        Returns:
            訂單字典 {symbol: quantity}
        """
        orders = {}
        
        # 計算需要買入/賣出的數量
        all_symbols = set(target_holdings.index) | set(current_holdings.keys())
        
        for symbol in all_symbols:
            target = target_holdings.get(symbol, 0.0)
            current = current_holdings.get(symbol, 0.0)
            diff = target - current
            
            if abs(diff) > 1e-10:
                orders[symbol] = diff
        
        return orders
    
    def run(self) -> BacktestResult:
        """
        執行回測。
        
        Returns:
            BacktestResult 包含淨值曲線、報酬率、績效指標等
        """
        # 初始化投資組合
        self._portfolio = Portfolio(initial_capital=self.initial_capital)
        
        timestamps = self._get_common_timestamps()
        
        # 第一個時間點：記錄初始狀態
        first_prices = self._get_prices_at(timestamps[0])
        self._portfolio.record_snapshot(timestamps[0], first_prices)
        
        # 從第二個時間點開始交易
        for i in range(1, len(timestamps)):
            current_ts = timestamps[i]
            prev_ts = timestamps[i - 1]
            
            # 1. 取得當前價格
            current_prices = self._get_prices_at(current_ts)
            
            # 2. 使用「前一期」信號計算目標權重（避免前視偏差）
            prev_signal = self._get_signal_at(prev_ts)
            
            # 跳過空信號
            if prev_signal.dropna().empty:
                self._portfolio.record_snapshot(current_ts, current_prices)
                continue
            
            # 3. 計算目標權重
            target_weights = self._calculate_target_weights(prev_signal)
            
            if target_weights.empty:
                self._portfolio.record_snapshot(current_ts, current_prices)
                continue
            
            # 4. 計算目標持倉
            total_value = self._portfolio.get_total_value(current_prices)
            target_holdings = self._calculate_target_holdings(
                target_weights, current_prices, total_value
            )
            
            # 5. 完全調倉模式：先平掉所有倉位
            if self.full_rebalance:
                for symbol, qty in list(self._portfolio.positions.items()):
                    if symbol in current_prices.index:
                        self._portfolio.execute_trade(
                            symbol, -qty, current_prices[symbol],
                            self.cost_rates, current_ts
                        )
            
            # 6. 生成並執行訂單
            orders = self._generate_orders(
                target_holdings, self._portfolio.positions
            )
            
            for symbol, qty in orders.items():
                if symbol in current_prices.index:
                    self._portfolio.execute_trade(
                        symbol, qty, current_prices[symbol],
                        self.cost_rates, current_ts
                    )
            
            # 7. 記錄快照
            self._portfolio.record_snapshot(current_ts, current_prices)
        
        # 構建結果
        return self._build_result()
    
    def _build_result(self) -> BacktestResult:
        """構建回測結果"""
        history_df = self._portfolio.get_history_df()
        trades_df = self._portfolio.get_trade_log_df()
        
        # 淨值曲線
        equity_curve = pd.Series(
            history_df['total_value'].values,
            index=pd.to_datetime(history_df['timestamp'], unit='ms'),
            name='equity'
        )
        
        # 報酬率序列
        returns = equity_curve.pct_change().dropna()
        returns.name = 'returns'
        
        # 計算績效指標
        metrics = calculate_metrics(returns)
        
        self._result = BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics,
            trades=trades_df,
            portfolio_history=history_df,
        )
        
        return self._result
    
    def summary(self) -> Dict[str, Any]:
        """
        返回回測摘要。
        
        Returns:
            包含績效指標的字典
        """
        if self._result is None:
            raise RuntimeError("Must call run() before summary()")
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': self._result.equity_curve.iloc[-1],
            'num_trades': len(self._result.trades),
            **self._result.metrics,
        }
    
    def plot_equity(self, figsize: tuple = (12, 6)) -> "matplotlib.figure.Figure":
        """
        繪製淨值曲線。
        
        Args:
            figsize: 圖片尺寸
            
        Returns:
            matplotlib Figure 物件
        """
        import matplotlib.pyplot as plt
        
        if self._result is None:
            raise RuntimeError("Must call run() before plot_equity()")
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # 淨值曲線
        self._result.equity_curve.plot(ax=axes[0], title='Equity Curve')
        axes[0].set_ylabel('Portfolio Value')
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        rolling_max = self._result.equity_curve.cummax()
        drawdown = (self._result.equity_curve - rolling_max) / rolling_max
        drawdown.plot(ax=axes[1], title='Drawdown', color='red')
        axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[1].set_ylabel('Drawdown')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
```

---

### 4.3 Metrics（績效指標）

```python
import numpy as np
import pandas as pd
from typing import Dict


def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 365.0 * 24,  # 假設小時級別
) -> Dict[str, float]:
    """
    計算績效指標。
    
    Args:
        returns: 報酬率序列
        risk_free_rate: 年化無風險利率
        periods_per_year: 每年的交易週期數
        
    Returns:
        績效指標字典
    """
    if returns.empty or returns.isna().all():
        return {
            'total_return': np.nan,
            'annual_return': np.nan,
            'annual_volatility': np.nan,
            'sharpe_ratio': np.nan,
            'sortino_ratio': np.nan,
            'calmar_ratio': np.nan,
            'max_drawdown': np.nan,
            'var_95': np.nan,
            'cvar_95': np.nan,
            'win_rate': np.nan,
            'profit_factor': np.nan,
        }
    
    returns = returns.dropna()
    
    # 基本統計
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    
    # 年化
    years = n_periods / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annual_volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
    
    # Sortino Ratio（只考慮下行風險）
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # VaR & CVaR (95%)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    
    # Win Rate & Profit Factor
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
    }
```

---

### 4.4 Utils（輔助函數）

```python
import numpy as np
import pandas as pd


def neutralize_weights(signals: pd.Series) -> pd.Series:
    """
    將信號轉換為美元中性權重。
    
    公式: (x - mean) / sum(|x - mean|)
    
    這確保：
    1. 多空權重之和為 0
    2. 總絕對權重為 1
    
    Args:
        signals: 原始信號值
        
    Returns:
        中性化後的權重
    """
    signals = signals.dropna()
    
    if len(signals) == 0:
        return pd.Series(dtype=float)
    
    mean = signals.mean()
    demeaned = signals - mean
    
    abs_sum = demeaned.abs().sum()
    if abs_sum == 0:
        return pd.Series(0.0, index=signals.index)
    
    return demeaned / abs_sum


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """安全除法，避免除以零"""
    if b == 0 or np.isnan(b):
        return default
    return a / b
```

---

## 5. 公開 API

### 5.1 `__init__.py`

```python
from .backtester import Backtester, BacktestResult
from .portfolio import Portfolio
from .metrics import calculate_metrics

__all__ = [
    'Backtester',
    'BacktestResult',
    'Portfolio',
    'calculate_metrics',
]
```

---

## 6. 使用範例

### 6.1 基本使用

```python
from factorium import AggBar
from factorium.backtest import Backtester

# 載入資料
agg = AggBar.from_parquet("data/multi_symbol.parquet")

# 構建策略因子
close = agg["close"]
momentum = (close / close.ts_shift(20)) - 1
signal = momentum.cs_rank().cs_zscore()

# 回測
bt = Backtester(
    prices=agg,
    signal=signal,
    entry_price="close",
    transaction_cost=(0.0003, 0.0003),  # 買賣各 0.03%
    initial_capital=10000,
    neutralization="market",
)

result = bt.run()

# 查看結果
print(bt.summary())
bt.plot_equity()
```

### 6.2 與 FactorEvaluator 串接

```python
from factorium import AggBar
from factorium.backtest import Backtester

agg = AggBar.from_parquet("data.parquet")
close = agg["close"]

# 構建因子
momentum = (close / close.ts_shift(20)) - 1

# 1. 先用 FactorEvaluator 評估因子特性
eval_result = momentum.eval(prices=close, periods=[1, 5, 10])
print(f"IC Mean: {eval_result['ic_mean']}")
print(f"Turnover: {eval_result['turnover_mean']}")

# 2. 再用 Backtester 評估實際報酬
signal = momentum.cs_rank()
bt = Backtester(prices=agg, signal=signal)
backtest_result = bt.run()
print(f"Sharpe: {backtest_result.metrics['sharpe_ratio']:.2f}")
```

---

## 7. 關鍵設計決策

### 7.1 避免前視偏差

```
T-1 時刻：計算信號
    ↓
T 時刻：使用 T-1 信號，以 T 時刻價格交易
    ↓
T+1 時刻：使用 T 信號...
```

### 7.2 美元中性權重計算

```python
# 原始信號: [0.8, 0.5, 0.3, 0.1]
# 去均值:   [0.375, 0.075, -0.125, -0.325]
# 歸一化:   [0.417, 0.083, -0.139, -0.361]
# 驗證: sum ≈ 0, sum(|w|) = 1
```

### 7.3 NaN 處理策略

| 情況 | 處理 |
|------|------|
| 信號全為 NaN | 跳過該時間點，維持現有持倉 |
| 部分信號為 NaN | 只交易有有效信號的標的 |
| 價格為 NaN | 跳過該標的的交易 |

---

## 8. 未來擴展

### Phase 1（核心功能）
- [x] 基本回測引擎
- [x] 美元中性策略
- [x] 績效指標計算
- [ ] 單元測試

### Phase 2（增強功能）
- [ ] 分組中性（Group-neutral）
- [ ] 部分調倉（只調整超過閾值的持倉）
- [ ] 槓桿控制
- [ ] 風險預算

### Phase 3（進階功能）
- [ ] 滑點模型
- [ ] 市場衝擊模型
- [ ] 多因子組合回測
- [ ] 走走停停（Walk-forward）優化

---

## 9. 測試計畫

### 9.1 單元測試

```python
# tests/backtest/test_portfolio.py
def test_execute_trade_buy():
    """測試買入交易"""
    
def test_execute_trade_sell():
    """測試賣出交易"""
    
def test_transaction_cost():
    """測試交易成本計算"""

# tests/backtest/test_backtester.py
def test_basic_backtest():
    """基本回測流程"""
    
def test_lookahead_bias():
    """驗證無前視偏差"""
    
def test_market_neutral():
    """驗證美元中性"""
```

### 9.2 整合測試

```python
def test_full_workflow():
    """完整工作流程：載入資料 → 構建因子 → 回測 → 評估"""
```

---

## 10. 參考

- [phandas/backtest.py](https://github.com/quantbai/phandas) - 原始參考實作
- Factorium FactorEvaluator - IC/IR 分析
- Factorium Factor - 因子運算
