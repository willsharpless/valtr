from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

from valtr.ir import (Binary, BinaryIROpKind, ConstBool, IntervalIR, IRId, IRNode, Nary, NaryKind, TemporalBinary,
                      TemporalUnary, Unary, UnaryIROpKind, Var)
from valtr.lexer import Span
from valtr.tl_parser import BinaryOp, BinaryOpKind, Identifier, Node, UnaryOp, UnaryOpKind


def _join_span(a: Span, b: Span) -> Span:
    return Span(start=a.start, end=b.end)


def _join_spans(spans: list[Span]) -> Span:
    # minimal join by indexes
    if not spans:
        # fallback; a zero-length synthetic span if needed
        return Span(start=spans[0].start, end=spans[0].end)  # only used when not empty
    start = min(spans, key=lambda s: s.start.index).start
    end = max(spans, key=lambda s: s.end.index).end
    return Span(start=start, end=end)


class IRBuilder:
    """
    - Hash-cons IR nodes (span excluded from the interning key)
    - Canonicalize Nary(AND/OR): flatten, dedupe, sort
    - Record a representative span on first creation (or computed if not given)
    """

    def __init__(self):
        self.nodes: list[IRNode] = []
        self._intern: Dict[tuple, IRId] = {}
        self._default_span = Span(start=None, end=None)  # type: ignore

    # ---------- intern primitive ----------
    def _get(self, key: tuple, node: IRNode) -> IRId:
        found = self._intern.get(key)
        if found is not None:
            return found
        nid = IRId(len(self.nodes))
        self._intern[key] = nid
        self.nodes.append(node)
        return nid

    def _span_or(self, span: Optional[Span]) -> Span:
        return span if span is not None else self._default_span

    # ---------- atoms ----------
    def var(self, name: str, span: Optional[Span] = None) -> IRId:
        key = ("Var", name)
        return self._get(key, Var(name=name, span=self._span_or(span)))

    def const_bool(self, value: bool, span: Optional[Span] = None) -> IRId:
        """
        Create a distinct ConstBool node per source occurrence.
        We *do* include span in the interning key for constants so they don't merge.
        """
        s = self._span_or(span)
        key = ("ConstBool", value, s.start.index, s.end.index)
        return self._get(key, ConstBool(value=value, span=s))

    # ---------- unary ----------
    def unary(self, kind: UnaryIROpKind, arg: IRId, span: Optional[Span] = None) -> IRId:
        key = ("Unary", kind, int(arg))
        node_span = self._span_or(span if span is not None else self.nodes[int(arg)].span)
        return self._get(key, Unary(kind=kind, arg=arg, span=node_span))

    def temporal_unary(
        self, kind: UnaryIROpKind, arg: IRId, iv: Optional[IntervalIR], span: Optional[Span] = None
    ) -> IRId:
        key = ("TemporalUnary", kind, int(arg), (iv.lo if iv else None, iv.hi if iv else None))
        node_span = self._span_or(span if span is not None else self.nodes[int(arg)].span)
        return self._get(key, TemporalUnary(kind=kind, arg=arg, interval=iv, span=node_span))

    # ---------- binary ----------
    def binary(self, kind: BinaryIROpKind, left: IRId, right: IRId, span: Optional[Span] = None) -> IRId:
        key = ("Binary", kind, int(left), int(right))
        if span is None:
            span = _join_span(self.nodes[int(left)].span, self.nodes[int(right)].span)
        return self._get(key, Binary(kind=kind, left=left, right=right, span=span))

    def temporal_binary(
        self, kind: BinaryIROpKind, left: IRId, right: IRId, iv: Optional[IntervalIR], span: Optional[Span] = None
    ) -> IRId:
        key = ("TemporalBinary", kind, int(left), int(right), (iv.lo if iv else None, iv.hi if iv else None))
        if span is None:
            span = _join_span(self.nodes[int(left)].span, self.nodes[int(right)].span)
        return self._get(key, TemporalBinary(kind=kind, left=left, right=right, interval=iv, span=span))

    # ---------- n-ary (canonicalized) ----------
    def nary(self, kind: NaryKind, args: Iterable[IRId], span: Optional[Span] = None) -> IRId:
        # flatten children with same op
        flat_ids: list[int] = []
        spans: list[Span] = []
        for rid in args:
            n = self.nodes[int(rid)]
            if isinstance(n, Nary) and n.kind == kind:
                for c in n.args:
                    flat_ids.append(int(c))
                    spans.append(self.nodes[int(c)].span)
            else:
                flat_ids.append(int(rid))
                spans.append(n.span)

        # dedupe + sort
        uniq_sorted = tuple(sorted(set(flat_ids)))

        # trivialities (neutral elements handled here if desired)
        if len(uniq_sorted) == 0:
            # If you want neutral constants: AND -> True, OR -> False
            if kind.name == "AND":
                return self.const_bool(True, span or (spans[0] if spans else None))
            if kind.name == "OR":
                return self.const_bool(False, span or (spans[0] if spans else None))
            raise ValueError("Empty N-ary with no neutral element")

        if span is None:
            # compute from children (not from sorted order; use min start / max end)
            span = _join_spans([self.nodes[i].span for i in uniq_sorted])

        if len(uniq_sorted) == 1:
            return IRId(uniq_sorted[0])

        key = ("Nary", kind, uniq_sorted)
        return self._get(key, Nary(kind=kind, args=[IRId(i) for i in uniq_sorted], span=span))
