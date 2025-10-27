from typing import Dict, Iterable, List, Optional, Protocol, Set, Tuple

from valtr.ir import (Binary, BinaryIROpKind, ConstBool, IntervalIR, IRId, IRNode, Nary, NaryKind, TemporalBinary,
                      TemporalUnary, Unary, UnaryIROpKind, Var)
from valtr.ir_builder import IRBuilder
from valtr.lexer import Span


def join_span(a: Span, b: Span) -> Span:
    return Span(start=a.start, end=b.end)


def join_spans(spans: List[Span], fallback: Optional[Span] = None) -> Span:
    if not spans:
        if fallback is None:
            raise ValueError("join_spans: empty spans and no fallback")
        return fallback
    start = min(spans, key=lambda s: s.start.index).start
    end = max(spans, key=lambda s: s.end.index).end
    return Span(start=start, end=end)


# ---------- Interval helpers ----------


def iv_tuple(iv: Optional[IntervalIR]) -> Tuple[Optional[int], Optional[int]]:
    return iv.lo if iv else None, iv.hi if iv else None


def iv_make(lo: Optional[int], hi: Optional[int]) -> Optional[IntervalIR]:
    if lo is None and hi is None:
        return None
    return IntervalIR(lo=lo, hi=hi)


# ---------- IR graph helpers ----------


def reachable(builder: IRBuilder, roots: Iterable[IRId] | IRId) -> Set[int]:
    if isinstance(roots, IRId):
        todo = [int(roots)]
    else:
        todo = [int(r) for r in roots]
    seen: Set[int] = set()
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        todo.extend(builder.nodes[i].children())
    return seen


# ---------- Small constructor helpers ----------


def make_true(dst: IRBuilder, span: Span) -> IRId:
    return dst.const_bool(True, span=span)


def make_and(dst: IRBuilder, args: Iterable[IRId], span: Span) -> IRId:
    return dst.nary(NaryKind.AND, args, span=span)


def make_or(dst: IRBuilder, args: Iterable[IRId], span: Span) -> IRId:
    return dst.nary(NaryKind.OR, args, span=span)


def make_globally(dst: IRBuilder, arg: IRId, iv: Optional[IntervalIR], span: Span) -> IRId:
    return dst.temporal_unary(UnaryIROpKind.GLOBALLY, arg, iv, span=span)


def make_until(dst: IRBuilder, left: IRId, right: IRId, iv: Optional[IntervalIR], span: Span) -> IRId:
    return dst.temporal_binary(BinaryIROpKind.UNTIL, left, right, iv, span=span)


class IRPass(Protocol):
    def run(self, root: IRId) -> tuple[IRId, "IRBuilder"]: ...


class IRRewriter:
    def __init__(self, src: IRBuilder):
        self.src = src
        self.dst = IRBuilder()
        self.memo: Dict[int, IRId] = {}

    def run(self, root: IRId) -> tuple[IRId, IRBuilder]:
        out = self.visit(root)
        return out, self.dst

    def visit(self, rid: IRId) -> IRId:
        i = int(rid)
        if i in self.memo:
            return self.memo[i]

        n = self.src.nodes[i]

        match n:
            case Var(name=name, span=span):
                out = self.rebuild_Var(name, span)

            case ConstBool(value=value, span=span):
                out = self.rebuild_ConstBool(value, span)

            case TemporalUnary(kind=kind, arg=arg, interval=iv, span=span):
                arg_id = self.visit(arg)
                out = self.rebuild_TemporalUnary(kind, arg_id, iv, span)

            case Unary(kind=kind, arg=arg, span=span):
                arg_id = self.visit(arg)
                out = self.rebuild_Unary(kind, arg_id, span)

            case TemporalBinary(kind=kind, left=left, right=right, interval=iv, span=span):
                l_id = self.visit(left)
                r_id = self.visit(right)
                out = self.rebuild_TemporalBinary(kind, l_id, r_id, iv, span)

            case Binary(kind=kind, left=left, right=right, span=span):
                l_id = self.visit(left)
                r_id = self.visit(right)
                out = self.rebuild_Binary(kind, l_id, r_id, span)

            case Nary(kind=kind, args=args, span=span):
                arg_ids = [self.visit(a) for a in args]
                out = self.rebuild_Nary(kind, arg_ids, span)

            case _:
                out = self.generic_visit(i, n)

        self.memo[i] = out
        return out

    # ---- Default rebuilders ----
    def rebuild_Var(self, name: str, span: Span) -> IRId:
        return self.dst.var(name, span)

    def rebuild_ConstBool(self, value: bool, span: Span) -> IRId:
        return self.dst.const_bool(value, span)

    def rebuild_Unary(self, kind: UnaryIROpKind, arg: IRId, span: Span) -> IRId:
        return self.dst.unary(kind, arg, span)

    def rebuild_TemporalUnary(self, kind: UnaryIROpKind, arg: IRId, iv: Optional[IntervalIR], span: Span) -> IRId:
        return self.dst.temporal_unary(kind, arg, iv, span)

    def rebuild_Binary(self, kind: BinaryIROpKind, left: IRId, right: IRId, span: Span) -> IRId:
        return self.dst.binary(kind, left, right, span)

    def rebuild_TemporalBinary(
        self, kind: BinaryIROpKind, left: IRId, right: IRId, iv: Optional[IntervalIR], span: Span
    ) -> IRId:
        return self.dst.temporal_binary(kind, left, right, iv, span)

    def rebuild_Nary(self, kind: NaryKind, args: List[IRId], span: Span) -> IRId:
        return self.dst.nary(kind, args, span)

    def generic_visit(self, i: int, n: IRNode) -> IRId:
        raise AssertionError(f"Unhandled IR node: {type(n).__name__}")
