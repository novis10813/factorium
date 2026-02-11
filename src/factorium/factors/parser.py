"""
Expression parser for Factor construction.

This module provides a parser for functional-style factor expressions,
enabling string-based factor construction similar to alpha101.
"""

from typing import Dict, Union, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Factor
from pyparsing import (
    Word,
    alphas,
    alphanums,
    nums,
    Optional,
    Group,
    Forward,
    infix_notation,
    OpAssoc,
    ParseException,
    Suppress,
    Combine,
    one_of,
)

from . import operators

if not TYPE_CHECKING:
    from .core import Factor


class FactorExpressionParser:
    """
    Parser for functional-style factor expressions.
    
    Supports:
    - Function calls: ts_delta(close, 20)
    - Variables: close, volume (resolved from context)
    - Numbers: 20, 3.14
    - Binary operators: +, -, *, / (with proper precedence)
    - Parentheses: (expression)
    
    Example:
        >>> parser = FactorExpressionParser()
        >>> result = parser.parse("ts_delta(close, 20) / ts_shift(close, 20)", 
        ...                       context={'close': close_factor})
    """
    
    def __init__(self):
        """Initialize the parser with grammar rules."""
        # Define basic tokens
        identifier = Word(alphas + "_", alphanums + "_")
        
        # Numbers (integers and floats)
        integer = Combine(Optional("-") + Word(nums))
        float_number = Combine(
            Optional("-") + Word(nums) + "." + Word(nums) +
            Optional(one_of("e E") + Optional(one_of("+ -")) + Word(nums))
        )
        number = float_number | integer
        
        # Function call: function_name(arg1, arg2, ...)
        function_call = Forward()
        
        # Expression (forward declaration for recursion)
        expression = Forward()
        
        # Argument list - each argument is its own Group to isolate results
        arg_list = Group(expression) + (Suppress(",") + Group(expression))[...]
        
        # Function call definition
        function_call <<= Group(
            identifier.set_results_name("func_name") + 
            Suppress("(") + 
            Optional(arg_list).set_results_name("args") + 
            Suppress(")")
        )
        
        # Variable or number - wrap in Group to isolate names
        atom = function_call | Group(identifier.set_results_name("variable")) | Group(number.set_results_name("number"))
        
        # Parenthesized expression
        paren_expr = Suppress("(") + expression + Suppress(")")
        
        # Primary factor (atom or parenthesized expression)
        factor = paren_expr | atom
        
        # Define operator precedence
        expression <<= infix_notation(
            factor,
            [
                (one_of("* /"), 2, OpAssoc.LEFT, self._make_binary_op),
                (one_of("+ -"), 2, OpAssoc.LEFT, self._make_binary_op),
            ],
        )
        
        self.parser = expression
    
    def _make_binary_op(self, tokens):
        """Action for infix operators - returns a dict structure"""
        # tokens[0] is a list: [left, op, right, op, right...]
        # Because we used OpAssoc.LEFT, tokens[0] contains the matched tokens
        matched = tokens[0] 
        res = matched[0]
        for i in range(1, len(matched), 2):
            op = matched[i]
            right = matched[i+1]
            res = {"type": "binary_op", "op": op, "left": res, "right": right}
        return res
    
    
    def _evaluate(self, node: Any, context: Dict[str, "Factor"]) -> Union["Factor", float, int]:
        """Evaluate a parsed expression node."""
        # Handle Factor objects directly
        from .core import Factor
        if isinstance(node, Factor):
            return node

        # Handle pyparsing ParseResults objects
        if hasattr(node, "as_dict"):
            try:
                node_dict = node.as_dict()
            except:
                node_dict = {}
            
            # If it's empty but has content, it might be a list-like ParseResults
            if not node_dict and hasattr(node, "__iter__") and not isinstance(node, (str, dict)):
                if len(node) == 1:
                    return self._evaluate(node[0], context)
            
            # Check if it's a binary operation (from infix_notation)
            node_type = node_dict.get("type")
            if node_type == "binary_op":
                op = node_dict["op"]
                left = self._evaluate(node_dict["left"], context)
                right = self._evaluate(node_dict["right"], context)
                print(f"DEBUG: binary_op {op}, left: {getattr(left, 'name', left)}, right: {getattr(right, 'name', right)}")
                
                if op == "+":
                    return operators.add(left, right)
                elif op == "-":
                    return operators.sub(left, right)
                elif op == "*":
                    return operators.mul(left, right)
                elif op == "/":
                    return operators.div(left, right)
                else:
                    raise ValueError(f"Unknown binary operator: {op}")
            
            # Check if it's a function call
            if "func_name" in node_dict:
                func_name = node_dict["func_name"]
                args_val = node_dict.get("args")
                
                if not hasattr(operators, func_name):
                    raise ValueError(f"Unknown function: {func_name}")
                
                op_func = getattr(operators, func_name)
                
                # Evaluate arguments
                if args_val is None:
                    eval_args = []
                elif hasattr(args_val, "__iter__") and not isinstance(args_val, (str, Factor)):
                    eval_args = [self._evaluate(arg, context) for arg in args_val]
                else:
                    eval_args = [self._evaluate(args_val, context)]
                
                return op_func(*eval_args)
            
            # Check if it's a variable
            if "variable" in node_dict:
                var_name = node_dict["variable"]
                if var_name not in context:
                    raise ValueError(f"Undefined variable: {var_name}")
                return context[var_name]
            
            # Check if it's a number
            if "number" in node_dict:
                num_str = str(node_dict["number"])
                try:
                    if "." in num_str or "e" in num_str.lower():
                        return float(num_str)
                    else:
                        return int(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")

        # Handle regular dictionaries (from _make_binary_op)
        if isinstance(node, dict):
            if node.get("type") == "binary_op":
                op = node["op"]
                left = self._evaluate(node["left"], context)
                right = self._evaluate(node["right"], context)
                
                if op == "+":
                    return operators.add(left, right)
                elif op == "-":
                    return operators.sub(left, right)
                elif op == "*":
                    return operators.mul(left, right)
                elif op == "/":
                    return operators.div(left, right)
                else:
                    raise ValueError(f"Unknown binary operator: {op}")
            
            if "func_name" in node:
                func_name = node["func_name"]
                args_list = node.get("args", [])
                if not hasattr(operators, func_name):
                    raise ValueError(f"Unknown function: {func_name}")
                op_func = getattr(operators, func_name)
                eval_args = [self._evaluate(arg, context) for arg in args_list] if isinstance(args_list, list) else [self._evaluate(args_list, context)]
                return op_func(*eval_args)
            
            if "variable" in node:
                var_name = node["variable"]
                if var_name not in context:
                    raise ValueError(f"Undefined variable: {var_name}")
                return context[var_name]
            
            if "number" in node:
                num_str = str(node["number"])
                try:
                    return float(num_str) if "." in num_str or "e" in num_str.lower() else int(num_str)
                except ValueError:
                    raise ValueError(f"Invalid number: {num_str}")

        # Handle lists/sequences
        if hasattr(node, "__iter__") and not isinstance(node, (str, Factor, dict)):
            node_list = list(node)
            if len(node_list) == 1:
                return self._evaluate(node_list[0], context)
            
            # If we have a list that didn't match anything above, it might be raw tokens of a function call
            # This shouldn't happen with our current grammar but let's be safe
            return self._evaluate(node_list[0], context)
        
        # Handle direct values
        if isinstance(node, (int, float, Factor)):
            return node
        
        if isinstance(node, str):
            if node in context:
                return context[node]
            # Try to resolve as number
            try:
                return float(node) if "." in node or "e" in node.lower() else int(node)
            except ValueError:
                # Check if it's a function name used without parentheses
                if hasattr(operators, node):
                    raise ValueError(f"Function '{node}' used without parentheses")
                raise ValueError(f"Undefined variable: {node}")
        
        # If we get here, it's an unexpected type
        raise ValueError(f"Unexpected node type: {type(node)}, value: {node}")
    
    def parse(self, expr: str, context: Dict[str, "Factor"]) -> "Factor":
        """
        Parse and evaluate a factor expression.
        
        Args:
            expr: Expression string (e.g., "ts_delta(close, 20) / ts_shift(close, 20)")
            context: Dictionary mapping variable names to Factor objects
            
        Returns:
            Factor: The resulting factor from the expression
            
        Raises:
            ParseException: If the expression cannot be parsed
            ValueError: If there's an error in evaluation (undefined variable, etc.)
        """
        try:
            parsed = self.parser.parse_string(expr, parse_all=True)
            result = self._evaluate(parsed, context)
            
            from .core import Factor
            if not isinstance(result, Factor):
                raise ValueError(f"Expression did not evaluate to a Factor, got {type(result)}")
            
            return result
        except ParseException as e:
            raise ValueError(f"Failed to parse expression '{expr}': {e}") from e

