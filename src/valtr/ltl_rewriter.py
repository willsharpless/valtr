from __future__ import annotations

from .ltl_builder import LTLBuilder
from .ltl_ir import And, Const, Expr, ExprId, Finally, Globally, Next, Not, Or, Release, Until, Var


class LTLRewriter:
    def __init__(self, src: LTLBuilder) -> None:
        self.src = src
        self.dst = LTLBuilder()
        self.memo: dict[int, ExprId] = {}

    def run(self, root: ExprId) -> tuple[ExprId, LTLBuilder]:
        return self.visit(root), self.dst

    def visit(self, expr_id: ExprId) -> ExprId:
        idx = int(expr_id)
        if idx in self.memo:
            return self.memo[idx]

        node = self.src.nodes[idx]
        if isinstance(node, Const):
            out = self.dst.const(node.value, node.origin)
        elif isinstance(node, Var):
            out = self.dst.var(node.name, node.origin)
        elif isinstance(node, Not):
            out = self.dst.not_(self.visit(node.arg), node.origin)
        elif isinstance(node, And):
            out = self.dst.and_((self.visit(arg) for arg in node.args), node.origin)
        elif isinstance(node, Or):
            out = self.dst.or_((self.visit(arg) for arg in node.args), node.origin)
        elif isinstance(node, Next):
            out = self.dst.next(self.visit(node.arg), node.interval, node.origin)
        elif isinstance(node, Finally):
            out = self.dst.finally_(self.visit(node.arg), node.interval, node.origin)
        elif isinstance(node, Globally):
            out = self.dst.globally(self.visit(node.arg), node.interval, node.origin)
        elif isinstance(node, Until):
            out = self.dst.until(self.visit(node.left), self.visit(node.right), node.interval, node.origin)
        elif isinstance(node, Release):
            out = self.dst.release(self.visit(node.left), self.visit(node.right), node.interval, node.origin)
        else:
            raise AssertionError(f"Unhandled LTL node: {type(node).__name__}")

        self.memo[idx] = out
        return out
