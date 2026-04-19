from .ir import (BinaryIROpKind, IntervalIR, IRId, NaryKind, UnaryIROpKind)
from .ir_builder import IRBuilder
from ..tl_parser import BinaryOp, Identifier, Interval, Node, UnaryOp


def _to_iv(iv_ast: Interval | None) -> IntervalIR | None:
    if iv_ast is None:
        return None
    return IntervalIR(lo=iv_ast.lo, hi=iv_ast.hi)


class Lowerer:
    def __init__(self, builder: IRBuilder | None = None):
        if builder is None:
            builder = IRBuilder()
        self.b = builder

    def lower(self, node: "Node") -> IRId:
        if isinstance(node, Identifier):
            return self.b.var(node.name, span=node.span)

        if isinstance(node, UnaryOp):
            arg = self.lower(node.operand)
            iv = _to_iv(node.interval)
            kind = UnaryIROpKind.from_token_type(node.op)
            if kind == UnaryIROpKind.NOT:
                return self.b.unary(kind, arg, span=node.span)
            return self.b.temporal_unary(kind, arg, iv, span=node.span)

        if isinstance(node, BinaryOp):
            left = self.lower(node.left)
            right = self.lower(node.right)
            iv = _to_iv(node.interval)
            k = BinaryIROpKind.from_token_type(node.op)
            if k in (BinaryIROpKind.UNTIL, BinaryIROpKind.RELEASE):
                return self.b.temporal_binary(k, left, right, iv, span=node.span)
            if k == BinaryIROpKind.IMPLIES:
                return self.b.binary(k, left, right, span=node.span)
            if k == BinaryIROpKind.AND:
                return self.b.nary(NaryKind.AND, (left, right), span=node.span)
            if k == BinaryIROpKind.OR:
                return self.b.nary(NaryKind.OR, (left, right), span=node.span)

        raise AssertionError(f"Unhandled AST node: {type(node).__name__}")
