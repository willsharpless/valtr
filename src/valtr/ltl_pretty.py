from __future__ import annotations

from .ltl_ir import And, Const, Expr, ExprId, Finally, Globally, Next, Not, Or, Release, Until, Var


def pretty_ltl(nodes: list[Expr], root: ExprId) -> str:
    return _pretty(nodes, root)


def _pretty(nodes: list[Expr], expr_id: ExprId) -> str:
    node = nodes[int(expr_id)]
    if isinstance(node, Const):
        return "T" if node.value else "F"
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Not):
        return f"!{_wrap(nodes, node.arg)}"
    if isinstance(node, And):
        return " && ".join(_wrap(nodes, arg) for arg in node.args)
    if isinstance(node, Or):
        return " || ".join(_wrap(nodes, arg) for arg in node.args)
    if isinstance(node, Next):
        return f"X {_wrap(nodes, node.arg)}"
    if isinstance(node, Finally):
        return f"F {_wrap(nodes, node.arg)}"
    if isinstance(node, Globally):
        return f"G {_wrap(nodes, node.arg)}"
    if isinstance(node, Until):
        return f"{_wrap(nodes, node.left)} U {_wrap(nodes, node.right)}"
    if isinstance(node, Release):
        return f"{_wrap(nodes, node.left)} R {_wrap(nodes, node.right)}"
    raise AssertionError(f"Unhandled LTL node: {type(node).__name__}")


def _wrap(nodes: list[Expr], expr_id: ExprId) -> str:
    node = nodes[int(expr_id)]
    if isinstance(node, (Const, Var)):
        return _pretty(nodes, expr_id)
    return f"({_pretty(nodes, expr_id)})"
