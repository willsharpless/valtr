from __future__ import annotations

from dataclasses import dataclass

from .ltl_ir import And, Const, Expr, ExprId, Globally, Or, Until


@dataclass(frozen=True, slots=True)
class UntilMatch:
    expr_id: ExprId
    left: ExprId
    right: ExprId


@dataclass(frozen=True, slots=True)
class GUMatch:
    expr_id: ExprId
    left: ExprId
    right: ExprId


@dataclass(frozen=True, slots=True)
class UGMatch:
    expr_id: ExprId
    left: ExprId
    guard: ExprId


def get_node(nodes: list[Expr], expr_id: ExprId) -> Expr:
    return nodes[int(expr_id)]


def conjuncts(expr_id: ExprId, nodes: list[Expr]) -> tuple[ExprId, ...]:
    node = get_node(nodes, expr_id)
    if isinstance(node, And):
        return node.args
    return (expr_id,)


def disjuncts(expr_id: ExprId, nodes: list[Expr]) -> tuple[ExprId, ...]:
    node = get_node(nodes, expr_id)
    if isinstance(node, Or):
        return node.args
    return (expr_id,)


def is_true(expr_id: ExprId, nodes: list[Expr]) -> bool:
    node = get_node(nodes, expr_id)
    return isinstance(node, Const) and node.value


def is_false(expr_id: ExprId, nodes: list[Expr]) -> bool:
    node = get_node(nodes, expr_id)
    return isinstance(node, Const) and not node.value


def is_propositional(expr_id: ExprId, nodes: list[Expr], memo: dict[int, bool] | None = None) -> bool:
    if memo is None:
        memo = {}
    idx = int(expr_id)
    if idx in memo:
        return memo[idx]
    node = get_node(nodes, expr_id)
    if isinstance(node, (Const,)):
        result = True
    elif node.__class__.__name__ == "Var":
        result = True
    elif node.__class__.__name__ == "Not":
        result = is_propositional(node.arg, nodes, memo)
    elif isinstance(node, And | Or):
        result = all(is_propositional(child, nodes, memo) for child in node.children())
    else:
        result = False
    memo[idx] = result
    return result


def is_plain_global_guard(expr_id: ExprId, nodes: list[Expr]) -> bool:
    node = get_node(nodes, expr_id)
    return isinstance(node, Globally) and node.interval is None and is_propositional(node.arg, nodes)


def get_until(expr_id: ExprId, nodes: list[Expr]) -> UntilMatch | None:
    node = get_node(nodes, expr_id)
    if isinstance(node, Until) and node.interval is None:
        return UntilMatch(expr_id=expr_id, left=node.left, right=node.right)
    return None


def get_gu(expr_id: ExprId, nodes: list[Expr]) -> GUMatch | None:
    node = get_node(nodes, expr_id)
    if not isinstance(node, Globally) or node.interval is not None:
        return None
    inner = get_node(nodes, node.arg)
    if isinstance(inner, Until) and inner.interval is None:
        return GUMatch(expr_id=expr_id, left=inner.left, right=inner.right)
    return None


def get_ug(expr_id: ExprId, nodes: list[Expr]) -> UGMatch | None:
    node = get_node(nodes, expr_id)
    if not isinstance(node, Until) or node.interval is not None:
        return None
    right = get_node(nodes, node.right)
    if isinstance(right, Globally) and right.interval is None:
        return UGMatch(expr_id=expr_id, left=node.left, guard=right.arg)
    return None


def is_fg_lowered(expr_id: ExprId, nodes: list[Expr]) -> bool:
    ug = get_ug(expr_id, nodes)
    return ug is not None and is_true(ug.left, nodes)


def replace_expr_ids(expr_ids: tuple[ExprId, ...], replacements: dict[int, ExprId], remove: set[int] | None = None) -> tuple[ExprId, ...]:
    remove = remove or set()
    out: list[ExprId] = []
    for expr_id in expr_ids:
        idx = int(expr_id)
        if idx in remove:
            continue
        out.append(replacements.get(idx, expr_id))
    return tuple(out)


def without_expr_ids(expr_ids: tuple[ExprId, ...], removed: set[int]) -> tuple[ExprId, ...]:
    return tuple(expr_id for expr_id in expr_ids if int(expr_id) not in removed)
