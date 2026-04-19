"""
Value Tree representation and transformation.

This module provides classes for representing value trees and transforming
temporal logic trees into value trees based on theoretical rules.
"""

from typing import Optional, Dict, Any, Tuple, List
import json
from .temporal_tree import TemporalLogicTree, TemporalLogicNode

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    FancyBboxPatch = None


class ValueNode:
    """A node in a value tree."""

    def __init__(
        self,
        operation: str,
        left: Optional["ValueNode"] = None,
        right: Optional["ValueNode"] = None,
        value: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a value node.

        Args:
            operation: The operation type (min, max, negation, etc.)
            left: Left child node
            right: Right child node
            value: Value for leaf nodes
            params: Additional parameters for the operation
        """
        self.operation = operation
        self.left = left
        self.right = right
        self.value = value
        self.params = params or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the node to a dictionary representation.

        Returns:
            Dictionary representation of the node and its children
        """
        result: Dict[str, Any] = {"operation": self.operation}

        if self.value is not None:
            result["value"] = self.value

        if self.params:
            result["params"] = self.params

        if self.left is not None:
            result["left"] = self.left.to_dict()

        if self.right is not None:
            result["right"] = self.right.to_dict()

        return result

    def to_string(self, indent: int = 0) -> str:
        """
        Convert the node to a string representation.

        Args:
            indent: Indentation level for pretty printing

        Returns:
            String representation of the node
        """
        prefix = "  " * indent
        if self.operation == "leaf":
            return f"{prefix}leaf: {self.value}"

        result = f"{prefix}{self.operation}"
        if self.params:
            result += f" (params: {self.params})"
        result += "\n"

        if self.left:
            result += self.left.to_string(indent + 1) + "\n"

        if self.right:
            result += self.right.to_string(indent + 1) + "\n"

        return result.rstrip()
    
    def to_latex(self) -> str:
        """
        Convert the node to a LaTeX representation.

        Returns:
            LaTeX string representation of the node
        """
        if self.operation == "leaf":
            return f"\\:{self.value}\\left( s_t \\right)"

        left_latex = self.left.to_latex() if self.left else ""
        right_latex = self.right.to_latex() if self.right else ""

        if self.operation == "min":
            return f"\\:\\: \\min\\left({left_latex}, {right_latex}\\right)"
        elif self.operation == "max":
            return f"\\:\\: \\max\\left({left_latex}, {right_latex}\\right)"
        elif self.operation == "negation":
            return f"\\:-{left_latex}"
        elif self.operation == "temporal_min":
            tb = self.params.get("time_bounds", "")
            tb = f"\\in [{tb[0]}, {tb[1]}]" if tb else ""
            return f"\\:\\: \\min_{{t {tb}}} {left_latex}"
        elif self.operation == "temporal_max":
            tb = self.params.get("time_bounds", "")
            tb = f"\\in [{tb[0]}, {tb[1]}]" if tb else ""
            return f"\\:\\: \\max_{{t {tb}}} {left_latex}"
        elif self.operation == "until":
            tb = self.params.get("time_bounds", "")
            tb = f"\\in [{tb[0]}, {tb[1]}]" if tb else ""
            # return f"\\left({left_latex}\\right) \\; U \\; \\left({right_latex}\\right)"
            return f"\\:\\: \\max \\left(\\min \\left({right_latex}, \\:\\: \\min_{{t {tb}}} {left_latex}\\right)\\right)"
        else:
            return f"{self.operation}\\left({left_latex}, {right_latex}\\right)"

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return self.to_string()


class ValueTree:
    """
    A value tree representation derived from a temporal logic tree.

    The value tree represents the temporal logic formula using value operations
    (min, max, negation) that can be evaluated with reachability analysis.
    """

    def __init__(self, root: ValueNode):
        """
        Initialize a value tree.

        Args:
            root: Root node of the value tree
        """
        self.root = root

    @classmethod
    def from_temporal_tree(cls, temporal_tree: TemporalLogicTree) -> "ValueTree":
        """
        Transform a temporal logic tree into a value tree using theoretical rules.

        Transformation rules:
        - AND (φ ∧ ψ) -> min(V(φ), V(ψ))
        - OR (φ ∨ ψ) -> max(V(φ), V(ψ))
        - NOT (¬φ) -> -V(φ)
        - IMPLIES (φ => ψ) -> max(-V(φ), V(ψ))
        - ALWAYS (□φ) -> min over time interval
        - EVENTUALLY (◇φ) -> max over time interval
        - UNTIL (φ U ψ) -> max over time of min(V(ψ), min(V(φ) over [0,t]))

        Args:
            temporal_tree: The temporal logic tree to transform

        Returns:
            A new ValueTree instance
        """
        root = cls._transform_node(temporal_tree.root)
        return cls(root)

    @classmethod
    def _transform_node(cls, node: TemporalLogicNode) -> ValueNode:
        """
        Transform a temporal logic node into a value node.

        Args:
            node: Temporal logic node to transform

        Returns:
            Transformed value node
        """
        if node.operator == "PREDICATE":
            # Leaf node - represents a predicate
            return ValueNode("leaf", value=node.value)

        elif node.operator == "AND":
            # AND -> min operation
            left = cls._transform_node(node.left) if node.left else None
            right = cls._transform_node(node.right) if node.right else None
            return ValueNode("min", left=left, right=right)

        elif node.operator == "OR":
            # OR -> max operation
            left = cls._transform_node(node.left) if node.left else None
            right = cls._transform_node(node.right) if node.right else None
            return ValueNode("max", left=left, right=right)

        elif node.operator == "NOT":
            # NOT -> negation
            child = cls._transform_node(node.left) if node.left else None
            return ValueNode("negation", left=child)

        elif node.operator == "IMPLIES":
            # IMPLIES (φ => ψ) -> max(-φ, ψ)
            left = cls._transform_node(node.left) if node.left else None
            right = cls._transform_node(node.right) if node.right else None
            neg_left = ValueNode("negation", left=left)
            return ValueNode("max", left=neg_left, right=right)

        elif node.operator == "ALWAYS":
            # ALWAYS -> min over time
            child = cls._transform_node(node.left) if node.left else None
            params = {}
            if node.time_bounds:
                params["time_bounds"] = node.time_bounds
            return ValueNode("temporal_min", left=child, params=params)

        elif node.operator == "EVENTUALLY":
            # EVENTUALLY -> max over time
            child = cls._transform_node(node.left) if node.left else None
            params = {}
            if node.time_bounds:
                params["time_bounds"] = node.time_bounds
            return ValueNode("temporal_max", left=child, params=params)

        elif node.operator == "UNTIL":
            # UNTIL -> special temporal composition
            left = cls._transform_node(node.left) if node.left else None
            right = cls._transform_node(node.right) if node.right else None
            return ValueNode("until", left=left, right=right)

        else:
            raise ValueError(f"Unknown operator: {node.operator}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tree to a dictionary representation.

        Returns:
            Dictionary representation of the tree
        """
        return {"type": "ValueTree", "root": self.root.to_dict()}

    def to_string(self) -> str:
        """
        Convert the tree to a string representation.

        Returns:
            String representation of the tree
        """
        return f"ValueTree:\n{self.root.to_string()}"

    def to_file(self, filepath: str, format: str = "json") -> None:
        """
        Save the tree to a file.

        Args:
            filepath: Path to the output file
            format: Output format ('json' or 'txt')
        """
        if format == "json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format == "txt":
            with open(filepath, "w") as f:
                f.write(self.to_string())
        else:
            raise ValueError(f"Unsupported format: {format}")

    def plot(self, filepath: Optional[str] = None, show: bool = True) -> None:
        """
        Plot the value tree structure.

        Args:
            filepath: Optional path to save the plot
            show: Whether to display the plot
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Calculate positions for nodes
        positions = self._calculate_positions(self.root)

        # Draw the tree
        self._draw_node(ax, self.root, positions)

        # Annotate bottom with formula
        formula = self.root.to_latex()
        formula_rend = f"$V(s) \\:\\:\\: = \\:\\:\\: \\max_{{\\pi}} {formula}$"
        ax.annotate(formula_rend, xy=(0.5, 0.05), xycoords="axes fraction", ha="center", fontsize=12)

        plt.title("Value Tree", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def _calculate_positions(
        self, node: ValueNode, x: float = 5.0, y: float = 9.0, level: int = 0
    ) -> Dict[int, Tuple[float, float]]:
        """
        Calculate positions for all nodes in the tree.

        Args:
            node: Current node
            x: X coordinate
            y: Y coordinate
            level: Current tree level

        Returns:
            Dictionary mapping node IDs to positions
        """
        positions = {id(node): (x, y)}

        if node.left or node.right:
            y_child = y - 1.5
            if node.left and node.right:
                # Both children
                x_left = x - 2.0 / (level + 1)
                x_right = x + 2.0 / (level + 1)
                positions.update(self._calculate_positions(node.left, x_left, y_child, level + 1))
                positions.update(self._calculate_positions(node.right, x_right, y_child, level + 1))
            elif node.left:
                # Only left child
                positions.update(self._calculate_positions(node.left, x, y_child, level + 1))

        return positions

    def _draw_node(
        self, ax, node: ValueNode, positions: Dict[int, Tuple[float, float]]
    ) -> None:
        """
        Draw a node and its children.

        Args:
            ax: Matplotlib axis
            node: Node to draw
            positions: Dictionary of node positions
        """
        x, y = positions[id(node)]

        # Draw edges to children
        if node.left:
            x_left, y_left = positions[id(node.left)]
            ax.plot([x, x_left], [y, y_left], "k-", linewidth=2, alpha=0.6, zorder=1)
            self._draw_node(ax, node.left, positions)

        if node.right:
            x_right, y_right = positions[id(node.right)]
            ax.plot([x, x_right], [y, y_right], "k-", linewidth=2, alpha=0.6, zorder=1)
            self._draw_node(ax, node.right, positions)

        # Draw node
        if node.operation == "leaf":
            color = "lightcoral"
            label = f"{node.value}"
        else:
            color = "lightyellow"
            label = node.operation
            if node.params:
                if "time_bounds" in node.params:
                    tb = node.params["time_bounds"]
                    label += f"\n[{tb[0]}, {tb[1]}]"

        bbox = FancyBboxPatch(
            (x - 0.4, y - 0.15),
            0.8,
            0.3,
            boxstyle="round,pad=0.05",
            edgecolor="black",
            facecolor=color,
            linewidth=2,
            alpha=0.9, # DOESNT HAVE AN EFFECT
            zorder=2
        )
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold")

    def __repr__(self) -> str:
        """Return a string representation of the tree."""
        return self.to_string()
