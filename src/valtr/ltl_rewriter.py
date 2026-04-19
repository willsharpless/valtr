from __future__ import annotations

from .ltl_builder import LTLBuilder
from .ltl_ir import And, Const, Expr, ExprId, Finally, Globally, Next, Not, Or, Origin, Release, Until, Var


class LTLRewriter:
    def __init__(self, src: LTLBuilder) -> None:
        self.src = src
        self.dst = LTLBuilder()
        self.memo: dict[int, ExprId] = {}
        self.changed = False

    def run(self, root: ExprId) -> tuple[ExprId, LTLBuilder]:
        return self.visit(root), self.dst

    def mark_changed(self) -> None:
        self.changed = True

    def origin_for(self, rule: str, *expr_ids: ExprId, original: Origin | None = None) -> Origin:
        origins = [self.dst.nodes[int(expr_id)].origin for expr_id in expr_ids]
        if original is not None:
            origins.append(original)
        return Origin.derived(rule, *origins, primary_span=original.primary_span if original else None)

    def visit(self, expr_id: ExprId) -> ExprId:
        idx = int(expr_id)
        if idx in self.memo:
            return self.memo[idx]

        node = self.src.nodes[idx]
        if isinstance(node, Const):
            out = self.rewrite_const(node)
        elif isinstance(node, Var):
            out = self.rewrite_var(node)
        elif isinstance(node, Not):
            out = self.rewrite_not(node, self.visit(node.arg))
        elif isinstance(node, And):
            out = self.rewrite_and(node, tuple(self.visit(arg) for arg in node.args))
        elif isinstance(node, Or):
            out = self.rewrite_or(node, tuple(self.visit(arg) for arg in node.args))
        elif isinstance(node, Next):
            out = self.rewrite_next(node, self.visit(node.arg))
        elif isinstance(node, Finally):
            out = self.rewrite_finally(node, self.visit(node.arg))
        elif isinstance(node, Globally):
            out = self.rewrite_globally(node, self.visit(node.arg))
        elif isinstance(node, Until):
            out = self.rewrite_until(node, self.visit(node.left), self.visit(node.right))
        elif isinstance(node, Release):
            out = self.rewrite_release(node, self.visit(node.left), self.visit(node.right))
        else:
            raise AssertionError(f"Unhandled LTL node: {type(node).__name__}")

        self.memo[idx] = out
        return out

    def rewrite_const(self, node: Const) -> ExprId:
        return self.dst.const(node.value, node.origin)

    def rewrite_var(self, node: Var) -> ExprId:
        return self.dst.var(node.name, node.origin)

    def rewrite_not(self, node: Not, arg: ExprId) -> ExprId:
        return self.dst.not_(arg, node.origin)

    def rewrite_and(self, node: And, args: tuple[ExprId, ...]) -> ExprId:
        return self.dst.and_(args, node.origin)

    def rewrite_or(self, node: Or, args: tuple[ExprId, ...]) -> ExprId:
        return self.dst.or_(args, node.origin)

    def rewrite_next(self, node: Next, arg: ExprId) -> ExprId:
        return self.dst.next(arg, node.interval, node.origin)

    def rewrite_finally(self, node: Finally, arg: ExprId) -> ExprId:
        return self.dst.finally_(arg, node.interval, node.origin)

    def rewrite_globally(self, node: Globally, arg: ExprId) -> ExprId:
        return self.dst.globally(arg, node.interval, node.origin)

    def rewrite_until(self, node: Until, left: ExprId, right: ExprId) -> ExprId:
        return self.dst.until(left, right, node.interval, node.origin)

    def rewrite_release(self, node: Release, left: ExprId, right: ExprId) -> ExprId:
        return self.dst.release(left, right, node.interval, node.origin)
