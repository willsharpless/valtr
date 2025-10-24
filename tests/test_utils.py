"""Tests for the utils module."""

import pytest
import numpy as np
from valtr.parser import STLParser
from valtr.value_tree import ValueTree, ValueNode
from valtr.dp import DynamicProgrammingSolver, evaluate_value_node

class TestDynamicProgrammingSolver:
    """Test cases for HJ reachability solver."""

    def test_create_solver(self):
        """Test creating a solver."""
        parser = STLParser()
        temporal_tree = parser.parse("p")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        solver = DynamicProgrammingSolver(value_tree)
        
        assert solver.value_tree == value_tree

    def test_solver_solve(self):
        """Test solve method (placeholder)."""
        parser = STLParser()
        temporal_tree = parser.parse("p")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        solver = DynamicProgrammingSolver(value_tree)
        
        result = solver.solve()
        assert "status" in result
        assert result["status"] == "placeholder"

    def test_solver_evaluate_at_state(self):
        """Test evaluate at state (placeholder)."""
        parser = STLParser()
        temporal_tree = parser.parse("p")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        solver = DynamicProgrammingSolver(value_tree)
        
        state = np.array([1.0, 2.0])
        result = solver.evaluate_at_state(state)
        assert isinstance(result, float)

    def test_solver_compute_value_function(self):
        """Test compute value function (placeholder)."""
        parser = STLParser()
        temporal_tree = parser.parse("p")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        solver = DynamicProgrammingSolver(value_tree)
        
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = solver.compute_value_function(grid)
        assert isinstance(result, np.ndarray)


class TestEvaluateValueNode:
    """Test cases for evaluate_value_node function."""

    def test_evaluate_leaf(self):
        """Test evaluating a leaf node."""
        node = ValueNode("leaf", value="p")
        state_values = {"p": 1.5}
        result = evaluate_value_node(node, state_values)
        assert result == 1.5

    def test_evaluate_leaf_missing_value(self):
        """Test evaluating leaf with missing value raises error."""
        node = ValueNode("leaf", value="p")
        state_values = {"q": 1.5}
        
        with pytest.raises(ValueError):
            evaluate_value_node(node, state_values)

    def test_evaluate_min(self):
        """Test evaluating min operation."""
        left = ValueNode("leaf", value="p")
        right = ValueNode("leaf", value="q")
        node = ValueNode("min", left=left, right=right)
        
        state_values = {"p": 1.5, "q": 2.0}
        result = evaluate_value_node(node, state_values)
        assert result == 1.5

    def test_evaluate_max(self):
        """Test evaluating max operation."""
        left = ValueNode("leaf", value="p")
        right = ValueNode("leaf", value="q")
        node = ValueNode("max", left=left, right=right)
        
        state_values = {"p": 1.5, "q": 2.0}
        result = evaluate_value_node(node, state_values)
        assert result == 2.0

    def test_evaluate_negation(self):
        """Test evaluating negation operation."""
        child = ValueNode("leaf", value="p")
        node = ValueNode("negation", left=child)
        
        state_values = {"p": 1.5}
        result = evaluate_value_node(node, state_values)
        assert result == -1.5

    def test_evaluate_complex_expression(self):
        """Test evaluating complex expression."""
        # min(p, max(q, -r))
        r_node = ValueNode("leaf", value="r")
        neg_r = ValueNode("negation", left=r_node)
        q_node = ValueNode("leaf", value="q")
        max_node = ValueNode("max", left=q_node, right=neg_r)
        p_node = ValueNode("leaf", value="p")
        root = ValueNode("min", left=p_node, right=max_node)
        
        state_values = {"p": 3.0, "q": 1.0, "r": 2.0}
        result = evaluate_value_node(root, state_values)
        # min(3.0, max(1.0, -2.0)) = min(3.0, 1.0) = 1.0
        assert result == 1.0

    def test_evaluate_temporal_operators(self):
        """Test evaluating temporal operators (simplified)."""
        child = ValueNode("leaf", value="p")
        node = ValueNode("temporal_min", left=child)
        
        state_values = {"p": 1.5}
        result = evaluate_value_node(node, state_values)
        # Simplified temporal evaluation just returns child value
        assert result == 1.5

    def test_evaluate_unknown_operation(self):
        """Test that unknown operation raises error."""
        node = ValueNode("unknown_op")
        state_values = {}
        
        with pytest.raises(ValueError):
            evaluate_value_node(node, state_values)
