"""
Basic usage examples for the valtr package.
"""

from valtr import STLParser, ValueTree

# Example 1: Simple predicate
print("=" * 60)
print("Example 1: Simple predicate")
print("=" * 60)
parser = STLParser()
tree = parser.parse("p")
print(tree)
print()

# Example 2: AND operator
print("=" * 60)
print("Example 2: AND operator (p & q)")
print("=" * 60)
tree = parser.parse("p&q")
print(tree)
print()

# Example 3: Complex formula with temporal operators
print("=" * 60)
print("Example 3: Complex formula ([]<>p & q)")
print("=" * 60)
tree = parser.parse("[]<>p&q")
print(tree)
print()

# Example 4: Transform to value tree
print("=" * 60)
print("Example 4: Transform to value tree")
print("=" * 60)
temporal_tree = parser.parse("(p&q)=>[]r")
print("Temporal tree:")
print(temporal_tree)
print()

value_tree = ValueTree.from_temporal_tree(temporal_tree)
print("Value tree:")
print(value_tree)
print()

# Example 5: Save trees to file
print("=" * 60)
print("Example 5: Save trees to files")
print("=" * 60)
temporal_tree.to_file("/tmp/temporal_tree.json", format="json")
temporal_tree.to_file("/tmp/temporal_tree.txt", format="txt")
value_tree.to_file("/tmp/value_tree.json", format="json")
value_tree.to_file("/tmp/value_tree.txt", format="txt")
print("Trees saved to /tmp/")
print()

# Example 6: Temporal operators with time bounds
print("=" * 60)
print("Example 6: Temporal operators with time bounds")
print("=" * 60)
tree = parser.parse("[]_[0,5]p")
print(tree)
value_tree = ValueTree.from_temporal_tree(tree)
print("Value tree:")
print(value_tree)
print()

# Example 7: Plot trees (saves to file without displaying)
print("=" * 60)
print("Example 7: Plot trees")
print("=" * 60)
try:
    temporal_tree = parser.parse("(p&q)|[]r")
    temporal_tree.plot(filepath="/tmp/temporal_tree.png", show=False)
    
    value_tree = ValueTree.from_temporal_tree(temporal_tree)
    value_tree.plot(filepath="/tmp/value_tree.png", show=False)
    print("Plots saved to /tmp/temporal_tree.png and /tmp/value_tree.png")
except ImportError as e:
    print(f"Plotting requires matplotlib: {e}")
print()

# Example 8: Evaluate value tree
print("=" * 60)
print("Example 8: Evaluate value tree")
print("=" * 60)
from valtr.utils import evaluate_value_node

temporal_tree = parser.parse("p&q|!r")
value_tree = ValueTree.from_temporal_tree(temporal_tree)

state_values = {"p": 2.5, "q": 1.0, "r": 3.0}
result = evaluate_value_node(value_tree.root, state_values)
print(f"Formula: p&q|!r")
print(f"State values: {state_values}")
print(f"Result: {result}")
print("(min(2.5, 1.0) = 1.0) max -3.0 = max(1.0, -3.0) = 1.0")
print()
