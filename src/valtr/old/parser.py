"""
Signal Temporal Logic (STL) Formula Parser.

This module provides functionality to parse STL formulas into a temporal logic tree.
"""

import re
from typing import Union, List, Tuple
from .temporal_tree import TemporalLogicTree, TemporalLogicNode


class STLParser:
    """Parser for Signal Temporal Logic (STL) formulas."""

    def __init__(self):
        """Initialize the STL parser."""
        self.operators = {
            "AND": "&",
            "OR": "|",
            "NOT": "!",
            "IMPLIES": "=>",
            "ALWAYS": "[]",
            "EVENTUALLY": "<>",
            "UNTIL": "U",
        }

    def parse(self, formula: str) -> TemporalLogicTree:
        """
        Parse an STL formula string into a TemporalLogicTree.

        Args:
            formula: STL formula string to parse

        Returns:
            TemporalLogicTree representation of the formula

        Examples:
            >>> parser = STLParser()
            >>> tree = parser.parse("[]<>p")
            >>> tree = parser.parse("(p & q) => <>r")
        """
        # Remove whitespace
        formula = formula.strip()

        # Parse the formula into a tree structure
        root = self._parse_expression(formula)

        return TemporalLogicTree(root)

    def _parse_expression(self, expr: str) -> TemporalLogicNode:
        """
        Parse an expression string into a TemporalLogicNode.

        Args:
            expr: Expression string to parse

        Returns:
            Root node of the parsed expression
        """
        expr = expr.strip()

        # Handle parentheses
        if expr.startswith("(") and expr.endswith(")"):
            # Check if these parentheses are balanced and wrapping the whole expression
            if self._is_fully_wrapped(expr):
                return self._parse_expression(expr[1:-1])

        # Check for binary operators (lowest precedence first)
        # Order: IMPLIES > OR > AND > UNTIL

        # Check for IMPLIES (=>)
        pos = self._find_operator(expr, "=>")
        if pos != -1:
            left = self._parse_expression(expr[:pos])
            right = self._parse_expression(expr[pos + 2 :])
            return TemporalLogicNode("IMPLIES", left=left, right=right)

        # Check for OR (|)
        pos = self._find_operator(expr, "|")
        if pos != -1:
            left = self._parse_expression(expr[:pos])
            right = self._parse_expression(expr[pos + 1 :])
            return TemporalLogicNode("OR", left=left, right=right)

        # Check for AND (&)
        pos = self._find_operator(expr, "&")
        if pos != -1:
            left = self._parse_expression(expr[:pos])
            right = self._parse_expression(expr[pos + 1 :])
            return TemporalLogicNode("AND", left=left, right=right)

        # Check for UNTIL (U)
        pos = self._find_operator(expr, "U")
        if pos != -1:
            left = self._parse_expression(expr[:pos])
            right = self._parse_expression(expr[pos + 1 :])
            return TemporalLogicNode("UNTIL", left=left, right=right)

        # Check for unary operators (highest precedence)
        # NOT (!)
        if expr.startswith("!"):
            child = self._parse_expression(expr[1:])
            return TemporalLogicNode("NOT", left=child)

        # Handle temporal operators with time bounds first
        # ALWAYS with time bound: []_[a,b]
        match = re.match(r"\[\]_\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\](.*)", expr)
        if match:
            t_min, t_max, rest = match.groups()
            child = self._parse_expression(rest)
            return TemporalLogicNode(
                "ALWAYS", left=child, time_bounds=(float(t_min), float(t_max))
            )

        # EVENTUALLY with time bound: <>_[a,b]
        match = re.match(r"<>_\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\](.*)", expr)
        if match:
            t_min, t_max, rest = match.groups()
            child = self._parse_expression(rest)
            return TemporalLogicNode(
                "EVENTUALLY", left=child, time_bounds=(float(t_min), float(t_max))
            )

        # ALWAYS ([])
        if expr.startswith("[]"):
            child = self._parse_expression(expr[2:])
            return TemporalLogicNode("ALWAYS", left=child)

        # EVENTUALLY (<>)
        if expr.startswith("<>"):
            child = self._parse_expression(expr[2:])
            return TemporalLogicNode("EVENTUALLY", left=child)

        # If we reach here, it should be a leaf node (atomic proposition or predicate)
        return TemporalLogicNode("PREDICATE", value=expr)

    def _find_operator(self, expr: str, op: str) -> int:
        """
        Find the position of an operator at the top level (not inside parentheses).

        Args:
            expr: Expression string
            op: Operator to find

        Returns:
            Position of operator, or -1 if not found at top level
        """
        depth = 0
        i = 0
        while i < len(expr):
            if expr[i] == "(":
                depth += 1
            elif expr[i] == ")":
                depth -= 1
            elif depth == 0 and expr[i : i + len(op)] == op:
                return i
            i += 1
        return -1

    def _is_fully_wrapped(self, expr: str) -> bool:
        """
        Check if an expression is fully wrapped in a single pair of parentheses.

        Args:
            expr: Expression string (should start with '(' and end with ')')

        Returns:
            True if the parentheses wrap the entire expression
        """
        if not (expr.startswith("(") and expr.endswith(")")):
            return False

        depth = 0
        for i, char in enumerate(expr):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth == 0 and i < len(expr) - 1:
                return False
        return True
