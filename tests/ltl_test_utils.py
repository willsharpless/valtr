from __future__ import annotations

from valtr.ltl_builder import LTLBuilder
from valtr.ltl_ir import And, Const, Expr, ExprId, Finally, Globally, Next, Not, Or, Release, Until, Var
from valtr.ltl_lowering import ASTToLTLLowerer
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser


def parse_ltl_spec(spec: str) -> tuple[LTLBuilder, ExprId]:
    tokens = list(TLLexer().tokenize(spec))
    ast = TLParser(tokens).parse()
    lowerer = ASTToLTLLowerer()
    root = lowerer.lower(ast)
    return lowerer.builder, root


def canonical_ltl_key(nodes: list[Expr], root: ExprId) -> tuple:
    return _canonical_ltl_key(nodes, root)


def assert_ltl_equivalent(
    actual_nodes: list[Expr],
    actual_root: ExprId,
    expected_spec: str,
    *,
    actual_rendered: str | None = None,
) -> None:
    expected_builder, expected_root = parse_ltl_spec(expected_spec)
    actual_key = canonical_ltl_key(actual_nodes, actual_root)
    expected_key = canonical_ltl_key(expected_builder.nodes, expected_root)
    assert actual_key == expected_key, (
        "LTL formulas are not structurally equivalent.\n"
        f"Actual:   {actual_rendered or actual_key}\n"
        f"Expected: {expected_spec}"
    )


def _canonical_ltl_key(nodes: list[Expr], expr_id: ExprId) -> tuple:
    node = nodes[int(expr_id)]

    if isinstance(node, Const):
        return ("Const", node.value)
    if isinstance(node, Var):
        return ("Var", node.name)
    if isinstance(node, Not):
        return ("Not", _canonical_ltl_key(nodes, node.arg))
    if isinstance(node, And):
        return ("And", tuple(sorted(_flatten_nary_key(nodes, node.args, And))))
    if isinstance(node, Or):
        return ("Or", tuple(sorted(_flatten_nary_key(nodes, node.args, Or))))
    if isinstance(node, Next):
        return ("Next", node.interval, _canonical_ltl_key(nodes, node.arg))
    if isinstance(node, Finally):
        return ("Finally", node.interval, _canonical_ltl_key(nodes, node.arg))
    if isinstance(node, Globally):
        return ("Globally", node.interval, _canonical_ltl_key(nodes, node.arg))
    if isinstance(node, Until):
        return (
            "Until",
            node.interval,
            _canonical_ltl_key(nodes, node.left),
            _canonical_ltl_key(nodes, node.right),
        )
    if isinstance(node, Release):
        return (
            "Release",
            node.interval,
            _canonical_ltl_key(nodes, node.left),
            _canonical_ltl_key(nodes, node.right),
        )
    raise AssertionError(f"Unhandled LTL node: {type(node).__name__}")


def _flatten_nary_key(nodes: list[Expr], expr_ids: tuple[ExprId, ...], kind: type[And] | type[Or]) -> list[tuple]:
    out: list[tuple] = []
    for expr_id in expr_ids:
        node = nodes[int(expr_id)]
        if isinstance(node, kind):
            out.extend(_flatten_nary_key(nodes, node.args, kind))
        else:
            out.append(_canonical_ltl_key(nodes, expr_id))
    return out
