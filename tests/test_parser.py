"""Tests for the STL parser module."""

import pytest
from valtr.parser import STLParser
from valtr.temporal_tree import TemporalLogicTree, TemporalLogicNode


class TestSTLParser:
    """Test cases for STL parser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = STLParser()

    def test_parse_simple_predicate(self):
        """Test parsing a simple predicate."""
        tree = self.parser.parse("p")
        assert tree.root.operator == "PREDICATE"
        assert tree.root.value == "p"

    def test_parse_and_operator(self):
        """Test parsing AND operator."""
        tree = self.parser.parse("p&q")
        assert tree.root.operator == "AND"
        assert tree.root.left.operator == "PREDICATE"
        assert tree.root.left.value == "p"
        assert tree.root.right.operator == "PREDICATE"
        assert tree.root.right.value == "q"

    def test_parse_or_operator(self):
        """Test parsing OR operator."""
        tree = self.parser.parse("p|q")
        assert tree.root.operator == "OR"
        assert tree.root.left.operator == "PREDICATE"
        assert tree.root.left.value == "p"
        assert tree.root.right.operator == "PREDICATE"
        assert tree.root.right.value == "q"

    def test_parse_not_operator(self):
        """Test parsing NOT operator."""
        tree = self.parser.parse("!p")
        assert tree.root.operator == "NOT"
        assert tree.root.left.operator == "PREDICATE"
        assert tree.root.left.value == "p"

    def test_parse_always_operator(self):
        """Test parsing ALWAYS operator."""
        tree = self.parser.parse("[]p")
        assert tree.root.operator == "ALWAYS"
        assert tree.root.left.operator == "PREDICATE"
        assert tree.root.left.value == "p"

    def test_parse_eventually_operator(self):
        """Test parsing EVENTUALLY operator."""
        tree = self.parser.parse("<>p")
        assert tree.root.operator == "EVENTUALLY"
        assert tree.root.left.operator == "PREDICATE"
        assert tree.root.left.value == "p"

    def test_parse_until_operator(self):
        """Test parsing UNTIL operator."""
        tree = self.parser.parse("pUq")
        assert tree.root.operator == "UNTIL"
        assert tree.root.left.operator == "PREDICATE"
        assert tree.root.left.value == "p"
        assert tree.root.right.operator == "PREDICATE"
        assert tree.root.right.value == "q"

    def test_parse_implies_operator(self):
        """Test parsing IMPLIES operator."""
        tree = self.parser.parse("p=>q")
        assert tree.root.operator == "IMPLIES"
        assert tree.root.left.operator == "PREDICATE"
        assert tree.root.left.value == "p"
        assert tree.root.right.operator == "PREDICATE"
        assert tree.root.right.value == "q"

    def test_parse_complex_formula(self):
        """Test parsing a complex formula."""
        tree = self.parser.parse("(p&q)=>[]r")
        assert tree.root.operator == "IMPLIES"
        assert tree.root.left.operator == "AND"
        assert tree.root.right.operator == "ALWAYS"

    def test_parse_nested_temporal_operators(self):
        """Test parsing nested temporal operators."""
        tree = self.parser.parse("[]<>p")
        assert tree.root.operator == "ALWAYS"
        assert tree.root.left.operator == "EVENTUALLY"
        assert tree.root.left.left.operator == "PREDICATE"

    def test_parse_with_parentheses(self):
        """Test parsing with parentheses."""
        tree = self.parser.parse("(p|q)&r")
        assert tree.root.operator == "AND"
        assert tree.root.left.operator == "OR"
        assert tree.root.right.operator == "PREDICATE"
        assert tree.root.right.value == "r"

    def test_parse_temporal_with_time_bounds(self):
        """Test parsing temporal operators with time bounds."""
        tree = self.parser.parse("[]_[0,5]p")
        assert tree.root.operator == "ALWAYS"
        assert tree.root.time_bounds == (0.0, 5.0)
        assert tree.root.left.operator == "PREDICATE"

        tree2 = self.parser.parse("<>_[1,3]q")
        assert tree2.root.operator == "EVENTUALLY"
        assert tree2.root.time_bounds == (1.0, 3.0)
