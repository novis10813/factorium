"""
Tests for Factor expression parser using dual verification strategy.

Forward path: Use existing Python method chaining (e.g., close.ts_delta(20))
Backward path: Use Factor.from_expression to verify same results
"""

import pytest
import pandas as pd
import numpy as np
from factorium import Factor, AggBar
from factorium.factors.parser import FactorExpressionParser

from tests.mixins.test_mathmixin import (
    sample_aggbar,  # noqa: F401
    factor_close,  # noqa: F401
    factor_open,  # noqa: F401
    assert_factor_equals_df,
)


# ==========================================
# Helper Functions
# ==========================================

def assert_expression_equals_method_chain(expr: str, context: dict, method_chain_result: Factor):
    """
    Verify that expression parsing produces the same result as method chaining.
    
    Args:
        expr: Expression string to parse
        context: Variable context for expression
        method_chain_result: Result from forward path (method chaining)
    """
    # Backward path: parse expression
    parsed_result = Factor.from_expression(expr, context)
    
    # Compare results
    assert_factor_equals_df(parsed_result, method_chain_result.data["factor"])


# ==========================================
# Test Cases: Time Series Operations
# ==========================================

def test_ts_delta_expression(factor_close):
    """Test ts_delta function in expression"""
    # Forward: method chaining
    forward_result = factor_close.ts_delta(5)
    
    # Backward: expression
    expr = "ts_delta(close, 5)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_ts_mean_expression(factor_close):
    """Test ts_mean function in expression"""
    forward_result = factor_close.ts_mean(10)
    expr = "ts_mean(close, 10)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_ts_std_expression(factor_close):
    """Test ts_std function in expression"""
    forward_result = factor_close.ts_std(10)
    expr = "ts_std(close, 10)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_ts_zscore_expression(factor_close):
    """Test ts_zscore function in expression"""
    forward_result = factor_close.ts_zscore(20)
    expr = "ts_zscore(close, 20)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_ts_shift_expression(factor_close):
    """Test ts_shift function in expression"""
    forward_result = factor_close.ts_shift(3)
    expr = "ts_shift(close, 3)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


# ==========================================
# Test Cases: Cross-Sectional Operations
# ==========================================

def test_rank_expression(factor_close):
    """Test rank function in expression"""
    forward_result = factor_close.rank()
    expr = "rank(close)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_mean_expression(factor_close):
    """Test mean function in expression"""
    forward_result = factor_close.mean()
    expr = "mean(close)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


# ==========================================
# Test Cases: Math Operations
# ==========================================

def test_abs_expression(factor_close):
    """Test abs function in expression"""
    forward_result = factor_close.abs()
    expr = "abs(close)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_log_expression(factor_close):
    """Test log function in expression"""
    forward_result = factor_close.log()
    expr = "log(close)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_sqrt_expression(factor_close):
    """Test sqrt function in expression"""
    forward_result = factor_close.sqrt()
    expr = "sqrt(close)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


# ==========================================
# Test Cases: Binary Operators
# ==========================================

def test_add_expression(factor_close, factor_open):
    """Test addition in expression"""
    forward_result = factor_close + factor_open
    expr = "add(close, open)"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_sub_expression(factor_close, factor_open):
    """Test subtraction in expression"""
    forward_result = factor_close - factor_open
    expr = "sub(close, open)"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_mul_expression(factor_close, factor_open):
    """Test multiplication in expression"""
    forward_result = factor_close * factor_open
    expr = "mul(close, open)"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_div_expression(factor_close, factor_open):
    """Test division in expression"""
    forward_result = factor_close / factor_open
    expr = "div(close, open)"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_add_scalar_expression(factor_close):
    """Test addition with scalar in expression"""
    forward_result = factor_close + 10
    expr = "add(close, 10)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_mul_scalar_expression(factor_close):
    """Test multiplication with scalar in expression"""
    forward_result = factor_close * 2.5
    expr = "mul(close, 2.5)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


# ==========================================
# Test Cases: Complex Expressions
# ==========================================

def test_momentum_expression(factor_close):
    """Test momentum calculation: ts_delta(close, 20) / ts_shift(close, 20)"""
    forward_result = factor_close.ts_delta(20) / factor_close.ts_shift(20)
    expr = "div(ts_delta(close, 20), ts_shift(close, 20))"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_zscore_normalized_expression(factor_close):
    """Test z-score normalization: ts_zscore(close, 20)"""
    forward_result = factor_close.ts_zscore(20)
    expr = "ts_zscore(close, 20)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_nested_operations_expression(factor_close):
    """Test nested operations: rank(ts_mean(close, 10))"""
    forward_result = factor_close.ts_mean(10).rank()
    expr = "rank(ts_mean(close, 10))"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_complex_arithmetic_expression(factor_close, factor_open):
    """Test complex arithmetic: (close + open) * 2"""
    forward_result = (factor_close + factor_open) * 2
    expr = "mul(add(close, open), 2)"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_operator_precedence_expression(factor_close, factor_open):
    """Test operator precedence: close + open * 2"""
    forward_result = factor_close + factor_open * 2
    expr = "add(close, mul(open, 2))"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_operator_precedence_with_division(factor_close, factor_open):
    """Test operator precedence: close * open / 2"""
    forward_result = factor_close * factor_open / 2
    expr = "div(mul(close, open), 2)"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


# ==========================================
# Test Cases: Expression Syntax (infix operators)
# ==========================================

def test_infix_add_expression(factor_close, factor_open):
    """Test infix addition: close + open"""
    forward_result = factor_close + factor_open
    expr = "close + open"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_infix_sub_expression(factor_close, factor_open):
    """Test infix subtraction: close - open"""
    forward_result = factor_close - factor_open
    expr = "close - open"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_infix_mul_expression(factor_close, factor_open):
    """Test infix multiplication: close * open"""
    forward_result = factor_close * factor_open
    expr = "close * open"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_infix_div_expression(factor_close, factor_open):
    """Test infix division: close / open"""
    forward_result = factor_close / factor_open
    expr = "close / open"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_infix_with_scalar_expression(factor_close):
    """Test infix with scalar: close * 2.5"""
    forward_result = factor_close * 2.5
    expr = "close * 2.5"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_infix_complex_expression(factor_close):
    """Test complex infix: ts_delta(close, 20) / ts_shift(close, 20)"""
    forward_result = factor_close.ts_delta(20) / factor_close.ts_shift(20)
    expr = "ts_delta(close, 20) / ts_shift(close, 20)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_infix_with_parentheses(factor_close, factor_open):
    """Test infix with parentheses: (close + open) * 2"""
    forward_result = (factor_close + factor_open) * 2
    expr = "(close + open) * 2"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


def test_infix_operator_precedence(factor_close, factor_open):
    """Test operator precedence: close + open * 2"""
    forward_result = factor_close + factor_open * 2
    expr = "close + open * 2"
    assert_expression_equals_method_chain(expr, {"close": factor_close, "open": factor_open}, forward_result)


# ==========================================
# Test Cases: Error Handling
# ==========================================

def test_undefined_variable_error(factor_close):
    """Test that undefined variable raises error"""
    expr = "ts_delta(unknown_var, 20)"
    with pytest.raises(ValueError, match="Undefined variable"):
        Factor.from_expression(expr, {"close": factor_close})


def test_unknown_function_error(factor_close):
    """Test that unknown function raises error"""
    expr = "unknown_func(close, 20)"
    with pytest.raises(ValueError, match="Unknown function"):
        Factor.from_expression(expr, {"close": factor_close})


def test_invalid_expression_syntax(factor_close):
    """Test that invalid syntax raises error"""
    expr = "ts_delta(close, 20"  # Missing closing parenthesis
    with pytest.raises(ValueError, match="Failed to parse"):
        Factor.from_expression(expr, {"close": factor_close})


# ==========================================
# Test Cases: Edge Cases
# ==========================================

def test_single_variable_expression(factor_close):
    """Test expression with just a variable"""
    forward_result = factor_close
    expr = "close"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_negative_number_expression(factor_close):
    """Test expression with negative number"""
    forward_result = factor_close + (-10)
    expr = "add(close, -10)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)


def test_float_number_expression(factor_close):
    """Test expression with float number"""
    forward_result = factor_close * 3.14159
    expr = "mul(close, 3.14159)"
    assert_expression_equals_method_chain(expr, {"close": factor_close}, forward_result)

