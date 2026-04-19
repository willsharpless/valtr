"""Live valtr package for the rewrite in progress.

This package currently keeps only the lexer/parser frontend in the active
namespace plus the new LTL rewrite IR. Legacy lowering, DAG, and solver code
lives under ``valtr.old``.
"""

from .lexer import LexError, Lexer, Position, Span, Token
from .ltl_builder import LTLBuilder
from .ltl_ir import (
    And,
    Const,
    Expr,
    ExprId,
    Finally,
    Globally,
    Interval as LTLInterval,
    Next,
    Not,
    Or,
    Origin,
    Release,
    SourceRef,
    Until,
    Var,
)
from .ltl_ir_graphviz import visualize_ltl_ir
from .ltl_lowering import ASTToLTLLowerer
from .ltl_pass_runner import LTLPassRunner, run_default_ltl_passes
from .ltl_pretty import pretty_ltl
from .ltl_rewriter import LTLRewriter
from .tl_lexer import TLLexer, TokenType
from .tl_parser import (
    BinaryOp,
    BinaryOpKind,
    ConstNode,
    Identifier,
    Interval,
    Node,
    ParseError,
    TLParser,
    UnaryOp,
    UnaryOpKind,
)

__version__ = "0.1.0"
__all__ = [
    "ASTToLTLLowerer",
    "And",
    "BinaryOp",
    "BinaryOpKind",
    "Const",
    "ConstNode",
    "Expr",
    "ExprId",
    "Finally",
    "Globally",
    "Identifier",
    "Interval",
    "LexError",
    "Lexer",
    "LTLBuilder",
    "LTLInterval",
    "LTLPassRunner",
    "LTLRewriter",
    "visualize_ltl_ir",
    "Node",
    "Next",
    "Not",
    "Or",
    "Origin",
    "ParseError",
    "Position",
    "Release",
    "run_default_ltl_passes",
    "SourceRef",
    "Span",
    "TLLexer",
    "TLParser",
    "Token",
    "TokenType",
    "Until",
    "UnaryOp",
    "UnaryOpKind",
    "Var",
    "pretty_ltl",
]
