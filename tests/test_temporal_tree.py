"""Tests for the temporal tree module."""

import pytest
import json
import tempfile
import os
from valtr.temporal_tree import TemporalLogicTree, TemporalLogicNode


class TestTemporalLogicNode:
    """Test cases for TemporalLogicNode."""

    def test_create_predicate_node(self):
        """Test creating a predicate node."""
        node = TemporalLogicNode("PREDICATE", value="p")
        assert node.operator == "PREDICATE"
        assert node.value == "p"
        assert node.left is None
        assert node.right is None

    def test_create_binary_node(self):
        """Test creating a binary operator node."""
        left = TemporalLogicNode("PREDICATE", value="p")
        right = TemporalLogicNode("PREDICATE", value="q")
        node = TemporalLogicNode("AND", left=left, right=right)
        assert node.operator == "AND"
        assert node.left == left
        assert node.right == right

    def test_create_unary_node(self):
        """Test creating a unary operator node."""
        child = TemporalLogicNode("PREDICATE", value="p")
        node = TemporalLogicNode("NOT", left=child)
        assert node.operator == "NOT"
        assert node.left == child
        assert node.right is None

    def test_node_to_dict(self):
        """Test converting node to dictionary."""
        left = TemporalLogicNode("PREDICATE", value="p")
        right = TemporalLogicNode("PREDICATE", value="q")
        node = TemporalLogicNode("AND", left=left, right=right)
        
        result = node.to_dict()
        assert result["operator"] == "AND"
        assert "left" in result
        assert "right" in result
        assert result["left"]["value"] == "p"
        assert result["right"]["value"] == "q"

    def test_node_to_string(self):
        """Test converting node to string."""
        node = TemporalLogicNode("PREDICATE", value="p")
        result = node.to_string()
        assert "p" in result

    def test_node_with_time_bounds(self):
        """Test node with time bounds."""
        child = TemporalLogicNode("PREDICATE", value="p")
        node = TemporalLogicNode("ALWAYS", left=child, time_bounds=(0.0, 5.0))
        assert node.time_bounds == (0.0, 5.0)
        result = node.to_dict()
        assert "time_bounds" in result
        assert result["time_bounds"] == [0.0, 5.0]


class TestTemporalLogicTree:
    """Test cases for TemporalLogicTree."""

    def test_create_tree(self):
        """Test creating a temporal logic tree."""
        root = TemporalLogicNode("PREDICATE", value="p")
        tree = TemporalLogicTree(root)
        assert tree.root == root

    def test_tree_to_dict(self):
        """Test converting tree to dictionary."""
        root = TemporalLogicNode("PREDICATE", value="p")
        tree = TemporalLogicTree(root)
        result = tree.to_dict()
        assert result["type"] == "TemporalLogicTree"
        assert "root" in result

    def test_tree_to_string(self):
        """Test converting tree to string."""
        root = TemporalLogicNode("PREDICATE", value="p")
        tree = TemporalLogicTree(root)
        result = tree.to_string()
        assert "TemporalLogicTree" in result
        assert "p" in result

    def test_tree_to_file_json(self):
        """Test saving tree to JSON file."""
        root = TemporalLogicNode("PREDICATE", value="p")
        tree = TemporalLogicTree(root)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            tree.to_file(filepath, format="json")
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert data["type"] == "TemporalLogicTree"
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_tree_to_file_txt(self):
        """Test saving tree to text file."""
        root = TemporalLogicNode("PREDICATE", value="p")
        tree = TemporalLogicTree(root)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            filepath = f.name
        
        try:
            tree.to_file(filepath, format="txt")
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
            assert "TemporalLogicTree" in content
            assert "p" in content
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_tree_to_file_invalid_format(self):
        """Test that invalid format raises error."""
        root = TemporalLogicNode("PREDICATE", value="p")
        tree = TemporalLogicTree(root)
        
        with pytest.raises(ValueError):
            tree.to_file("test.xyz", format="invalid")

    def test_tree_plot_no_show(self):
        """Test plotting tree without showing (for CI environments)."""
        root = TemporalLogicNode("PREDICATE", value="p")
        tree = TemporalLogicTree(root)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            filepath = f.name
        
        try:
            tree.plot(filepath=filepath, show=False)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
