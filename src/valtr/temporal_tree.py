"""
Temporal Logic Tree representation.

This module provides classes for representing and manipulating temporal logic trees.
"""

from typing import Optional, Dict, Any, Tuple, List
import json

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    FancyBboxPatch = None


class TemporalLogicNode:
    """A node in a temporal logic tree."""

    def __init__(
        self,
        operator: str,
        left: Optional["TemporalLogicNode"] = None,
        right: Optional["TemporalLogicNode"] = None,
        value: Optional[str] = None,
        time_bounds: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize a temporal logic node.

        Args:
            operator: The operator type (AND, OR, NOT, ALWAYS, EVENTUALLY, UNTIL, PREDICATE)
            left: Left child node (or only child for unary operators)
            right: Right child node (for binary operators)
            value: Value for leaf nodes (predicates)
            time_bounds: Optional time bounds (min, max) for temporal operators
        """
        self.operator = operator
        self.left = left
        self.right = right
        self.value = value
        self.time_bounds = time_bounds

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the node to a dictionary representation.

        Returns:
            Dictionary representation of the node and its children
        """
        result: Dict[str, Any] = {"operator": self.operator}

        if self.value is not None:
            result["value"] = self.value

        if self.time_bounds is not None:
            result["time_bounds"] = list(self.time_bounds)

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
        if self.operator == "PREDICATE":
            return f"{prefix}{self.value}"

        result = f"{prefix}{self.operator}"
        if self.time_bounds:
            result += f"[{self.time_bounds[0]}, {self.time_bounds[1]}]"
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

        # Leaves / predicates
        if self.operator == "PREDICATE":
            # \: is fine in mathtext; ensures a small space before symbol
            return rf"\:{self.value}"

        # Mathtext-safe operator symbols (no usetex required)
        operator_map = {
            "AND": r"\wedge", # "\\land",
            "OR": r"\vee", # "\\lor",
            "NOT": r"\neg", # "\\lnot",
            "ALWAYS": "□", # "\\Box",
            "EVENTUALLY": "◇", # "\\Diamond",
            "UNTIL": r"\, U \,",
        }

        op = operator_map.get(self.operator, self.operator)

        sub = ""
        if getattr(self, "time_bounds", None):
            a, b = self.time_bounds
            sub = rf"_{{[{a}, {b}]}}"

        def P(expr: str) -> str:
            return rf"\left(\:{expr}\right)"

        if self.operator in {"NOT", "ALWAYS", "EVENTUALLY"}:
            if self.left and not self.right:
                return rf"{op}{sub}" + self.left.to_latex()
            else:
                return rf"{op}{sub}" + P(self.left.to_latex())

        if self.operator in {"AND", "OR", "UNTIL"}:
            return P(self.left.to_latex() + rf"\:\:{op}{sub}\:\:" + self.right.to_latex())

        else:
            raise ValueError("Unknown Operator")

    def __repr__(self) -> str:
        """Return a string representation of the node."""
        return self.to_string()


class TemporalLogicTree:
    """A tree representation of a temporal logic formula."""

    def __init__(self, root: TemporalLogicNode):
        """
        Initialize a temporal logic tree.

        Args:
            root: Root node of the tree
        """
        self.root = root

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tree to a dictionary representation.

        Returns:
            Dictionary representation of the tree
        """
        return {"type": "TemporalLogicTree", "root": self.root.to_dict()}

    def to_string(self) -> str:
        """
        Convert the tree to a string representation.

        Returns:
            String representation of the tree
        """
        return f"TemporalLogicTree:\n{self.root.to_string()}"
    
    def to_latex(self) -> str:
        """
        Convert the tree to a LaTeX representation.

        Returns:
            LaTeX string representation of the tree
        """
        return self.root.to_latex()

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
        Plot the tree structure.

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
        formula = self.to_latex()
        formula_rend = rf"$\varphi \, := \, {formula}$"   # raw string; thin spaces
        ax.annotate(formula_rend, xy=(0.5, 0.05), xycoords="axes fraction",
                    ha="center", fontsize=12)

        plt.title("Temporal Logic Tree", fontsize=16, fontweight="bold")
        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def _calculate_positions(
        self, node: TemporalLogicNode, x: float = 5.0, y: float = 9.0, level: int = 0
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
        self, ax, node: TemporalLogicNode, positions: Dict[int, Tuple[float, float]]
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
        if node.operator == "PREDICATE":
            color = "lightgreen"
            label = node.value
        else:
            color = "lightblue"
            label = node.operator
            if node.time_bounds:
                label += f"\n[{node.time_bounds[0]}, {node.time_bounds[1]}]"

        bbox = FancyBboxPatch(
            (x - 0.4, y - 0.15),
            0.8,
            0.3,
            boxstyle="round,pad=0.05",
            edgecolor="black",
            facecolor=color,
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold")

    def __repr__(self) -> str:
        """Return a string representation of the tree."""
        return self.to_string()
