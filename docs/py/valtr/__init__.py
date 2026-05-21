"""
Valtr: A tool for generating value trees from signal temporal logic formulas.

This package provides:
- STL formula parsing
- Temporal logic tree representation
- Value tree generation and transformation
- Utilities for hj_reachability integration
"""

from .parser import STLParser
from .temporal_tree import TemporalLogicTree
from .value_tree import ValueTree

__version__ = "0.1.0"
__all__ = ["STLParser", "TemporalLogicTree", "ValueTree"]
