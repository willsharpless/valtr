from __future__ import annotations

from .ltl_builder import LTLBuilder, ast_origin
from .ltl_ir import ExprId, Interval
from .tl_parser import BinaryOp, BinaryOpKind, ConstNode, Identifier, Node, UnaryOp, UnaryOpKind


def _to_interval(interval) -> Interval | None:
    if interval is None:
        return None
    return Interval(lo=interval.lo, hi=interval.hi)


class ASTToLTLLowerer:
    def __init__(self, builder: LTLBuilder | None = None) -> None:
        self.builder = builder if builder is not None else LTLBuilder()

    def lower(self, node: Node) -> ExprId:
        return self._lower(node)

    def _lower(self, node: Node) -> ExprId:
        origin = ast_origin(id(node), node.span)

        if isinstance(node, ConstNode):
            return self.builder.const(node.value, origin)

        if isinstance(node, Identifier):
            return self.builder.var(node.name, origin)

        if isinstance(node, UnaryOp):
            arg = self._lower(node.operand)
            interval = _to_interval(node.interval)
            if node.op == UnaryOpKind.NOT:
                return self.builder.not_(arg, origin)
            if node.op == UnaryOpKind.FINALLY:
                return self.builder.finally_(arg, interval, origin)
            if node.op == UnaryOpKind.GLOBALLY:
                return self.builder.globally(arg, interval, origin)
            if node.op == UnaryOpKind.NEXT:
                return self.builder.next(arg, interval, origin)
            raise AssertionError(f"Unhandled unary op: {node.op}")

        if isinstance(node, BinaryOp):
            left = self._lower(node.left)
            right = self._lower(node.right)
            interval = _to_interval(node.interval)
            if node.op == BinaryOpKind.AND:
                return self.builder.and_((left, right), origin)
            if node.op == BinaryOpKind.OR:
                return self.builder.or_((left, right), origin)
            if node.op == BinaryOpKind.IMPLIES:
                return self.builder.implies(left, right, origin)
            if node.op == BinaryOpKind.UNTIL:
                return self.builder.until(left, right, interval, origin)
            if node.op == BinaryOpKind.RELEASE:
                return self.builder.release(left, right, interval, origin)
            raise AssertionError(f"Unhandled binary op: {node.op}")

        raise AssertionError(f"Unhandled AST node: {type(node).__name__}")
