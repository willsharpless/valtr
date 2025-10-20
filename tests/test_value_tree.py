"""Tests for the value tree module."""

import pytest
import json
import tempfile
import os
from valtr.parser import STLParser
from valtr.value_tree import ValueTree, ValueNode


class TestValueNode:
    """Test cases for ValueNode."""

    def test_create_leaf_node(self):
        """Test creating a leaf node."""
        node = ValueNode("leaf", value="p")
        assert node.operation == "leaf"
        assert node.value == "p"

    def test_create_binary_node(self):
        """Test creating a binary operation node."""
        left = ValueNode("leaf", value="p")
        right = ValueNode("leaf", value="q")
        node = ValueNode("min", left=left, right=right)
        assert node.operation == "min"
        assert node.left == left
        assert node.right == right

    def test_node_to_dict(self):
        """Test converting node to dictionary."""
        left = ValueNode("leaf", value="p")
        right = ValueNode("leaf", value="q")
        node = ValueNode("min", left=left, right=right)
        
        result = node.to_dict()
        assert result["operation"] == "min"
        assert "left" in result
        assert "right" in result

    def test_node_to_string(self):
        """Test converting node to string."""
        node = ValueNode("leaf", value="p")
        result = node.to_string()
        assert "leaf" in result
        assert "p" in result


class TestValueTree:
    """Test cases for ValueTree."""

    def test_create_value_tree(self):
        """Test creating a value tree."""
        root = ValueNode("leaf", value="p")
        tree = ValueTree(root)
        assert tree.root == root

    def test_from_temporal_tree_predicate(self):
        """Test transforming a simple predicate."""
        parser = STLParser()
        temporal_tree = parser.parse("p")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "leaf"
        assert value_tree.root.value == "p"

    def test_from_temporal_tree_and(self):
        """Test transforming AND operator."""
        parser = STLParser()
        temporal_tree = parser.parse("p&q")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "min"
        assert value_tree.root.left.operation == "leaf"
        assert value_tree.root.right.operation == "leaf"

    def test_from_temporal_tree_or(self):
        """Test transforming OR operator."""
        parser = STLParser()
        temporal_tree = parser.parse("p|q")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "max"
        assert value_tree.root.left.operation == "leaf"
        assert value_tree.root.right.operation == "leaf"

    def test_from_temporal_tree_not(self):
        """Test transforming NOT operator."""
        parser = STLParser()
        temporal_tree = parser.parse("!p")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "negation"
        assert value_tree.root.left.operation == "leaf"

    def test_from_temporal_tree_always(self):
        """Test transforming ALWAYS operator."""
        parser = STLParser()
        temporal_tree = parser.parse("[]p")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "temporal_min"
        assert value_tree.root.left.operation == "leaf"

    def test_from_temporal_tree_eventually(self):
        """Test transforming EVENTUALLY operator."""
        parser = STLParser()
        temporal_tree = parser.parse("<>p")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "temporal_max"
        assert value_tree.root.left.operation == "leaf"

    def test_from_temporal_tree_implies(self):
        """Test transforming IMPLIES operator."""
        parser = STLParser()
        temporal_tree = parser.parse("p=>q")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "max"
        assert value_tree.root.left.operation == "negation"
        assert value_tree.root.right.operation == "leaf"

    def test_from_temporal_tree_until(self):
        """Test transforming UNTIL operator."""
        parser = STLParser()
        temporal_tree = parser.parse("pUq")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "until"
        assert value_tree.root.left.operation == "leaf"
        assert value_tree.root.right.operation == "leaf"

    def test_from_temporal_tree_complex(self):
        """Test transforming complex formula."""
        parser = STLParser()
        temporal_tree = parser.parse("[]<>p&q")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "min"
        assert value_tree.root.left.operation == "temporal_min"

    def test_from_temporal_tree_with_time_bounds(self):
        """Test transforming temporal operators with time bounds."""
        parser = STLParser()
        temporal_tree = parser.parse("[]_[0,5]p")
        value_tree = ValueTree.from_temporal_tree(temporal_tree)
        
        assert value_tree.root.operation == "temporal_min"
        assert "time_bounds" in value_tree.root.params
        assert value_tree.root.params["time_bounds"] == (0.0, 5.0)

    def test_value_tree_to_dict(self):
        """Test converting value tree to dictionary."""
        root = ValueNode("leaf", value="p")
        tree = ValueTree(root)
        result = tree.to_dict()
        
        assert result["type"] == "ValueTree"
        assert "root" in result

    def test_value_tree_to_string(self):
        """Test converting value tree to string."""
        root = ValueNode("leaf", value="p")
        tree = ValueTree(root)
        result = tree.to_string()
        
        assert "ValueTree" in result
        assert "leaf" in result

    def test_value_tree_to_file_json(self):
        """Test saving value tree to JSON file."""
        root = ValueNode("leaf", value="p")
        tree = ValueTree(root)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            tree.to_file(filepath, format="json")
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert data["type"] == "ValueTree"
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_value_tree_to_file_txt(self):
        """Test saving value tree to text file."""
        root = ValueNode("leaf", value="p")
        tree = ValueTree(root)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            filepath = f.name
        
        try:
            tree.to_file(filepath, format="txt")
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
            assert "ValueTree" in content
            assert "leaf" in content
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_value_tree_plot_no_show(self):
        """Test plotting value tree without showing (for CI environments)."""
        root = ValueNode("leaf", value="p")
        tree = ValueTree(root)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            filepath = f.name
        
        try:
            tree.plot(filepath=filepath, show=False)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
