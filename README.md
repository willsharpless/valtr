# valtr

A Python package for generating value trees from Signal Temporal Logic (STL) formulas. 

`valtr` parses STL formulas, creates temporal logic tree representations, and transforms them into value trees based on theoretical rules. These value trees can be used with Hamilton-Jacobi (HJ) reachability analysis tools.

## Features

- **STL Formula Parsing**: Parse Signal Temporal Logic formulas into temporal logic trees
- **Temporal Logic Trees**: Rich tree representation with support for:
  - Boolean operators: AND (`&`), OR (`|`), NOT (`!`), IMPLIES (`=>`)
  - Temporal operators: ALWAYS (`[]`), EVENTUALLY (`<>`), UNTIL (`U`)
  - Time-bounded temporal operators: `[]_[a,b]`, `<>_[a,b]`
- **Value Tree Transformation**: Automatic transformation from temporal logic to value trees using theoretical rules:
  - AND → min operation
  - OR → max operation
  - NOT → negation
  - ALWAYS → temporal min
  - EVENTUALLY → temporal max
- **Multiple Output Formats**:
  - Dictionary representation
  - String/text representation
  - JSON and text file export
  - Visualization plots (with matplotlib)
- **HJ Reachability Integration**: Utilities for solving value trees with HJ reachability analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/willsharpless/valtr.git
cd valtr

# Install the package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from valtr import STLParser, ValueTree

# Parse an STL formula
parser = STLParser()
temporal_tree = parser.parse("[]<>p & q")

# Display the temporal logic tree
print(temporal_tree)

# Transform to value tree
value_tree = ValueTree.from_temporal_tree(temporal_tree)
print(value_tree)

# Save to file
temporal_tree.to_file("temporal_tree.json", format="json")
value_tree.to_file("value_tree.json", format="json")

# Plot the trees
temporal_tree.plot(filepath="temporal_tree.png", show=False)
value_tree.plot(filepath="value_tree.png", show=False)
```

## STL Syntax

### Boolean Operators
- `p & q` - AND (conjunction)
- `p | q` - OR (disjunction)
- `!p` - NOT (negation)
- `p => q` - IMPLIES (implication)

### Temporal Operators
- `[]p` - ALWAYS (globally)
- `<>p` - EVENTUALLY (finally)
- `p U q` - UNTIL

### Time-Bounded Operators
- `[]_[a,b]p` - ALWAYS over time interval [a,b]
- `<>_[a,b]p` - EVENTUALLY over time interval [a,b]

### Complex Formulas
- `(p & q) => []r` - If p and q, then always r
- `[]<>p` - Always eventually p (infinitely often)
- `<>[]p` - Eventually always p (stabilization)
- `[]_[0,5]p & <>_[2,3]q` - Combined temporal constraints

## Usage Examples

### Example 1: Parse and Display

```python
from valtr import STLParser

parser = STLParser()
tree = parser.parse("(p & q) | []r")
print(tree)
```

Output:
```
TemporalLogicTree:
OR
  AND
    p
    q
  ALWAYS
    r
```

### Example 2: Transform to Value Tree

```python
from valtr import STLParser, ValueTree

# Parse STL formula
parser = STLParser()
temporal_tree = parser.parse("p & q")

# Transform to value tree
value_tree = ValueTree.from_temporal_tree(temporal_tree)
print(value_tree)
```

Output:
```
ValueTree:
min
  leaf: p
  leaf: q
```

### Example 3: Evaluate Value Tree

```python
from valtr import STLParser, ValueTree
from valtr.utils import evaluate_value_node

# Create value tree
parser = STLParser()
temporal_tree = parser.parse("p & q | !r")
value_tree = ValueTree.from_temporal_tree(temporal_tree)

# Evaluate with specific values
state_values = {"p": 2.5, "q": 1.0, "r": 3.0}
result = evaluate_value_node(value_tree.root, state_values)
print(f"Result: {result}")  # Output: 1.0
```

### Example 4: Time-Bounded Operators

```python
from valtr import STLParser, ValueTree

parser = STLParser()
tree = parser.parse("[]_[0,5]p & <>_[2,3]q")
print(tree)

value_tree = ValueTree.from_temporal_tree(tree)
print(value_tree)
```

### Example 5: HJ Reachability Integration

```python
from valtr import STLParser, ValueTree
from valtr.utils import DynamicProgrammingSolver
import numpy as np

# Create value tree
parser = STLParser()
temporal_tree = parser.parse("[]p")
value_tree = ValueTree.from_temporal_tree(temporal_tree)

# Initialize solver (placeholder for actual HJ reachability)
solver = DynamicProgrammingSolver(value_tree)
result = solver.solve()
print(result)
```

## Transformation Rules

The package transforms temporal logic trees into value trees using the following theoretical rules:

| Temporal Logic | Value Tree Operation |
|---------------|---------------------|
| `p & q` (AND) | `min(V(p), V(q))` |
| `p | q` (OR) | `max(V(p), V(q))` |
| `!p` (NOT) | `-V(p)` |
| `p => q` (IMPLIES) | `max(-V(p), V(q))` |
| `[]p` (ALWAYS) | `min over time` |
| `<>p` (EVENTUALLY) | `max over time` |
| `p U q` (UNTIL) | Special composition |

## API Reference

### STLParser

```python
parser = STLParser()
tree = parser.parse(formula: str) -> TemporalLogicTree
```

### TemporalLogicTree

```python
tree.to_dict() -> Dict[str, Any]
tree.to_string() -> str
tree.to_file(filepath: str, format: str = "json")
tree.plot(filepath: Optional[str] = None, show: bool = True)
```

### ValueTree

```python
value_tree = ValueTree.from_temporal_tree(temporal_tree: TemporalLogicTree)
value_tree.to_dict() -> Dict[str, Any]
value_tree.to_string() -> str
value_tree.to_file(filepath: str, format: str = "json")
value_tree.plot(filepath: Optional[str] = None, show: bool = True)
```

### Utilities

```python
from valtr.utils import evaluate_value_node, DynamicProgrammingSolver

# Evaluate value node
result = evaluate_value_node(node: ValueNode, state_values: Dict[str, float])

# HJ Reachability solver (placeholder for integration)
solver = DynamicProgrammingSolver(value_tree: ValueTree)
result = solver.solve()
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=valtr --cov-report=term-missing
```

### Code Style

The project follows standard Python coding conventions:
- PEP 8 style guide
- Type hints for better code documentation
- Comprehensive docstrings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## References

- Signal Temporal Logic (STL): A formal language for specifying properties of continuous-time signals
- Hamilton-Jacobi Reachability: A method for computing reachable sets and value functions
- Value Functions: Numerical representations used in optimal control and reachability analysis
