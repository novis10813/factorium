# Backtest 模块潜在问题分析

## 严重问题

### 1. 现金不足检查缺失 ⚠️ **严重**

**位置**: `portfolio.py:execute_trade()`

**问题描述**:
- 在买入交易时，代码没有检查现金是否足够支付交易金额和手续费
- 如果现金不足，`self.cash` 会变成负数，导致不现实的回测结果

**当前代码**:
```python
if is_buy:
    self.cash -= trade_value + cost  # 没有检查 cash 是否足够
```

**影响**:
- 允许"借钱"交易，导致回测结果不准确
- 在杠杆交易场景中可能被误认为是正常行为，但实际上没有杠杆限制

**建议修复**:
```python
if is_buy:
    required = trade_value + cost
    if self.cash < required:
        # 选项1: 拒绝交易
        return
        # 选项2: 部分执行
        # max_qty = (self.cash - cost) / price
        # quantity = min(quantity, max_qty)
    self.cash -= required
```

---

### 2. 价格缺失导致持仓不匹配 ⚠️ **中等**

**位置**: `backtester.py:run()`

**问题描述**:
- 在 `_calculate_target_holdings()` 中，如果某个 symbol 在 `target_weights` 中存在，但在 `current_prices` 中不存在，该 symbol 会被排除
- 但在 `_generate_orders()` 中，如果该 symbol 在 `target_holdings` 中但不在 `current_prices` 中，订单会被跳过（第 221 行检查）
- 这可能导致目标持仓和实际持仓不一致

**当前代码**:
```python
# 第 150-152 行：只使用共同 symbols
common_symbols = weights.index.intersection(prices.index)
weights = weights.loc[common_symbols]
prices = prices.loc[common_symbols]

# 第 221 行：如果 symbol 不在 prices 中，跳过交易
if symbol in current_prices.index:
    self._portfolio.execute_trade(...)
```

**影响**:
- 持仓可能无法完全匹配目标权重
- 在价格数据不完整的情况下，回测结果可能不准确

**建议修复**:
- 在计算目标持仓时，记录哪些 symbols 被排除
- 在生成订单时，明确处理价格缺失的情况
- 或者在验证阶段就检查信号和价格的一致性

---

## 逻辑问题

### 3. 持仓数量计算精度问题 ⚠️ **轻微**

**位置**: `backtester.py:_calculate_target_holdings()`

**问题描述**:
- 使用 `target_values / prices` 计算持仓数量
- 如果价格非常小（接近 0），可能导致持仓数量非常大
- 浮点数精度可能导致持仓数量不准确

**当前代码**:
```python
target_values = weights * total_value
target_quantities = target_values / prices  # 没有检查价格是否合理
```

**影响**:
- 极端价格可能导致异常大的持仓数量
- 可能触发数值溢出或精度问题

**建议修复**:
```python
target_values = weights * total_value
target_quantities = target_values / prices

# 添加合理性检查
if prices.min() < 1e-10:  # 或使用 POSITION_EPSILON
    # 处理异常价格
    pass
```

---

### 4. full_rebalance 模式下的现金管理 ⚠️ **轻微**

**位置**: `backtester.py:run()`

**问题描述**:
- 在 `full_rebalance=True` 时，先卖出所有持仓，然后执行新订单
- 如果卖出后现金增加，然后买入新持仓，可能导致持仓价值超过初始资本
- 这是预期的行为（允许使用卖出获得的现金），但需要确保逻辑正确

**当前代码**:
```python
if self.full_rebalance:
    for symbol, qty in list(self._portfolio.positions.items()):
        if symbol in current_prices.index:
            self._portfolio.execute_trade(
                symbol, -qty, float(current_prices[symbol]), ...
            )

orders = self._generate_orders(target_holdings, self._portfolio.positions)
# 执行新订单...
```

**影响**:
- 如果逻辑正确，这是预期行为
- 但如果卖出和买入之间存在价格变化，可能导致不一致

**建议修复**:
- 确保在同一个时间点使用相同的价格进行卖出和买入
- 当前代码已经这样做了，所以这个问题可能不存在

---

## 指标计算问题

### 5. annual_return 在 years=0 时的处理 ⚠️ **轻微**

**位置**: `metrics.py:calculate_metrics()`

**问题描述**:
- 当 `years == 0` 时，返回 `0.0` 可能不太合理
- 应该返回 `np.nan` 表示无法计算

**当前代码**:
```python
annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
```

**影响**:
- 在数据不足的情况下，可能误导用户

**建议修复**:
```python
annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
```

---

### 6. sortino_ratio 在没有下行收益时的处理 ⚠️ **轻微**

**位置**: `metrics.py:calculate_metrics()`

**问题描述**:
- 如果没有下行收益，返回 `np.inf` 可能不太合理
- 应该考虑返回一个有限值或 `np.nan`

**当前代码**:
```python
sortino_ratio = np.inf if excess_return > 0 else 0.0
```

**影响**:
- `np.inf` 在图表和报告中可能显示异常

**建议修复**:
```python
if len(downside_returns) == 0:
    # 没有下行风险，但也没有上行收益时，返回 NaN
    sortino_ratio = np.nan if excess_return == 0 else np.inf
else:
    downside_std = float(downside_returns.std() * np.sqrt(periods_per_year))
    sortino_ratio = safe_divide(excess_return, downside_std, default=0.0)
```

---

## 边界情况

### 7. 持仓数量接近零的处理 ⚠️ **轻微**

**位置**: `portfolio.py:execute_trade()`

**问题描述**:
- 当 `abs(new_qty) < POSITION_EPSILON` 时，持仓被移除
- 但如果持仓数量在 `POSITION_EPSILON` 附近波动，可能导致频繁的持仓创建和删除

**当前代码**:
```python
if abs(new_qty) < POSITION_EPSILON:
    self.positions.pop(symbol, None)
else:
    self.positions[symbol] = new_qty
```

**影响**:
- 可能导致交易日志中出现大量小额交易
- 计算开销增加

**建议**:
- 当前实现已经合理，但可以考虑在生成订单时就过滤掉小于阈值的订单

---

### 8. 空信号处理 ⚠️ **轻微**

**位置**: `backtester.py:run()`

**问题描述**:
- 当信号为空时，代码会跳过该时间点，只记录快照
- 这是合理的，但可能导致某些时间点没有交易记录

**当前代码**:
```python
if prev_signal.dropna().empty:
    self._portfolio.record_snapshot(current_ts, current_prices)
    continue
```

**影响**:
- 如果连续多个时间点都没有信号，持仓会保持不变
- 这是预期行为，但需要确保逻辑正确

**建议**:
- 当前实现已经合理

---

## 总结

### 优先级修复建议

1. **高优先级**:
   - 修复现金不足检查（问题 #1）

2. **中优先级**:
   - 修复价格缺失导致的持仓不匹配（问题 #2）

3. **低优先级**:
   - 改进指标计算的边界情况处理（问题 #5, #6）
   - 添加持仓数量计算的合理性检查（问题 #3）

### 测试建议

建议添加以下测试用例：

1. **现金不足测试**:
   - 测试当现金不足以支付买入交易时的行为
   - 验证交易是否被拒绝或部分执行

2. **价格缺失测试**:
   - 测试当信号中的 symbol 在价格数据中不存在时的行为
   - 验证持仓是否与目标权重一致

3. **边界情况测试**:
   - 测试极端价格（接近 0 或非常大）的情况
   - 测试连续多个时间点没有信号的情况
