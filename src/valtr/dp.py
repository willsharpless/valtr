"""
Utility functions for valtr package.

This module provides utility functions for working with value trees,
including integration with hj_reachability solver.
"""

from typing import Optional, Callable, Dict, Any
import numpy as np
from .value_tree import ValueTree, ValueNode
from .dynamics.point import Point

try:
    import jax.numpy as jnp
    import hj_reachability as hj
    import hj_reachability.dynamics as dynamics
    # from jax.config import config
    # config.update("jax_enable_x64", True)
except ImportError:
    raise ImportError(
        "Dynamic-Programming requires hj_reachability (+jax). Install with: pip install valtr[dp]"
    )

class DynamicProgrammingSolver:
    """
    Solver for value trees using HJ reachability analysis.

    This class provides methods to evaluate value trees using
    Hamilton-Jacobi reachability analysis techniques.
    """

    def __init__(self, 
                    value_tree: ValueTree, 
                    dynamics: Optional[dynamics.Dynamics] = Point,
                    dynamics_kwargs: Optional[Dict[str, Any]] = None,
                    time_start: float = -1.0,
                    time_end: float = 0.,
                    time_step: float = 0.1,
                    grid: Optional[np.ndarray] = None,
                    grid_bounds_lo: Optional[np.ndarray] = None,
                    grid_bounds_hi: Optional[np.ndarray] = None,
                    grid_lengths: Optional[np.ndarray] = None,
                    uniform_grid_bound: float = 2.,
                    uniform_grid_length: int = 51,
    ):
        """
        Initialize the solver with a value tree.

        Args:
            value_tree: The value tree to solve
            dynamics: System dynamics class (default: Point dynamics)
            dynamics_kwargs: Additional arguments for dynamics initialization
            time_start: Start time for analysis
            time_end: End time for analysis
            time_step: Time step for analysis
            grid: Optional state space grid
            grid_bounds_lo: Lower bounds for uniform grid (if grid not provided)
            grid_bounds_hi: Upper bounds for uniform grid (if grid not provided)
            grid_lengths: Number of grid points per dimension (if grid not provided)
            uniform_grid_bound: Bound for uniform grid (if grid not provided)
            uniform_grid_length: Number of points per dimension for uniform grid (if grid not provided)
        """
        self.value_tree = value_tree
        self.times = np.linspace(time_start, time_end, int((time_end - time_start) / time_step) + 1)
        self.dynamics = dynamics(**(dynamics_kwargs or {}))
        if grid is None:
            grid_bounds_lo = grid_bounds_lo if grid_bounds_lo is not None else -uniform_grid_bound * np.ones(self.dynamics.dim)
            grid_bounds_hi = grid_bounds_hi if grid_bounds_hi is not None else uniform_grid_bound * np.ones(self.dynamics.dim)
            grid_lengths = grid_lengths if grid_lengths is not None else uniform_grid_length * np.ones(self.dynamics.dim, dtype=int)
            self.grid = hj.Grid.from_grid_parameters(
                lower_bounds=grid_bounds_lo,
                upper_bounds=grid_bounds_hi,
                sizes=grid_lengths,
            )
        else:
            self.grid = grid

        ## Define the initial values for each predicate in the value tree
        self.initial_values = {}
        for node in self.value_tree.get_leaves():
            
            # RANDOM VALUES FOR TESTING FIXME
            random_center = np.random.uniform(-uniform_grid_bound, uniform_grid_bound, size=(self.dynamics.dim,))
            random_radius = np.random.uniform(0.05, uniform_grid_bound / 4)
            random_reward = np.linalg.norm(self.grid - random_center, axis=-1) - random_radius
            dummy_penalty = -2.*random_reward.min() + 0*random_reward
            self.initial_values[node.name] = {"reward": random_reward, "penalty": dummy_penalty}

        self.results = {}
        for node in self.value_tree.get_all_nodes():
            self.results[node.name] = None

    def solve(
        self,
        state_space: Optional[np.ndarray] = None,
        dynamics: Optional[Callable] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Solve the value tree using dynamic programming (hj_reachability.py).

        Args:
            state_space: State space grid for reachability analysis
            dynamics: System dynamics function
            **kwargs: Additional solver parameters

        Returns:
            Dictionary containing solution results
        """

        ## Iterate thru computation graph of value tree
        for node in self.value_tree.post_order_traversal():
            
            ## If leaf node, set initial value
            if node.type == "leaf":
                initial_value = self.initial_values[node.name]
                self.results[node.name] = self.compute_value_function(initial_value, node.operation)

            else:
                ## Get operation for non-leaf nodes
                children_values = [self.results[child.name] for child in node.get_children()] # FIXME
                assert all([cv is not None for cv in children_values]), \
                    f"Child values not computed for node {node.name}: {[child.name for child in node.get_children() if self.results[child.name] is None]}"
                self.results[node.name] = self.compute_value_function(initial_value, node.operation, children_values=children_values)

        return self.results

    def compute_value_function(
        self, initial_value: np.ndarray, operation: str, children_values: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute the value function over the defined grid. To do this, defines the correct post-processing based on the operation type.

        Args:
            initial_value: Initial value function (boundary condition)
            operation: Operation to perform (e.g., "temporal_max" (REACH), "temporal_min" (AVOID), "temporal_max_min" (REACH-AVOID))
            children_values: values from child nodes (if applicable)

        Returns:
            Value function over the grid
        """

        # Placeholder implementation
        return np.zeros(initial_value.shape)

def visualize_value_function(
    value_function: np.ndarray,
    grid: np.ndarray,
    filepath: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize a computed value function.

    Args:
        value_function: The value function to visualize
        grid: State space grid
        filepath: Optional path to save the plot
        show: Whether to display the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")

    # Handle different dimensionalities
    if len(value_function.shape) == 1:
        # 1D case
        plt.figure(figsize=(10, 6))
        plt.plot(grid, value_function, linewidth=2)
        plt.xlabel("State")
        plt.ylabel("Value")
        plt.title("Value Function")
        plt.grid(True)

    elif len(value_function.shape) == 2:
        # 2D case
        plt.figure(figsize=(10, 8))
        plt.contourf(grid[0], grid[1], value_function, levels=20, cmap="viridis")
        plt.colorbar(label="Value")
        plt.xlabel("State 1")
        plt.ylabel("State 2")
        plt.title("Value Function")
        plt.contour(grid[0], grid[1], value_function, levels=[0], colors="red", linewidths=2)

    else:
        raise ValueError("Can only visualize 1D or 2D value functions")

    plt.tight_layout()

    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

def evaluate_value_node(
    node: ValueNode, state_values: Dict[str, float], time: float = 0.0
) -> float:
    """
    Recursively evaluate a value node given predicate values.

    This provides a simple evaluation of the value tree structure
    without full HJ reachability analysis.

    Args:
        node: The value node to evaluate
        state_values: Dictionary mapping predicate names to their values
        time: Current time (for temporal operators)

    Returns:
        Evaluated value
    """
    if node.operation == "leaf":
        # Return the predicate value
        predicate_name = node.value
        if predicate_name not in state_values:
            raise ValueError(f"No value provided for predicate: {predicate_name}")
        return state_values[predicate_name]

    elif node.operation == "min":
        # Minimum of children
        left_val = evaluate_value_node(node.left, state_values, time) if node.left else float("inf")
        right_val = (
            evaluate_value_node(node.right, state_values, time) if node.right else float("inf")
        )
        return min(left_val, right_val)

    elif node.operation == "max":
        # Maximum of children
        left_val = (
            evaluate_value_node(node.left, state_values, time) if node.left else float("-inf")
        )
        right_val = (
            evaluate_value_node(node.right, state_values, time) if node.right else float("-inf")
        )
        return max(left_val, right_val)

    elif node.operation == "negation":
        # Negation of child
        if node.left:
            return -evaluate_value_node(node.left, state_values, time)
        return 0.0

    elif node.operation in ["temporal_min", "temporal_max", "until"]:
        # For temporal operators, we'd need time-series data
        # This is a simplified placeholder
        if node.left:
            return evaluate_value_node(node.left, state_values, time)
        return 0.0

    else:
        raise ValueError(f"Unknown operation: {node.operation}")
