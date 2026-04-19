from typing import List, Optional

from .ir import BinaryIROpKind, IntervalIR, IRId, Nary, NaryKind, TemporalUnary, UnaryIROpKind, Var, ConstBool, \
    Unary, TemporalBinary, Binary
from .ir_builder import IRBuilder
from .ir_rewriter import IRPass, IRRewriter
from ..lexer import Span


def join_spans(spans: List[Span], fallback: Span) -> Span:
    if not spans:
        return fallback
    start = min(spans, key=lambda s: s.start.index).start
    end = max(spans, key=lambda s: s.end.index).end
    return Span(start=start, end=end)


# ============================================================
# Pass 1: Finally(φ, iv)  =>  Until(True, φ, iv)
# ============================================================


class PassFinallyToUntil(IRRewriter):
    """
    Rewrites every TemporalUnary FINALLY into a TemporalBinary UNTIL
    with a True constant on the left, preserving intervals and spans.
    """

    def visit(self, rid: IRId) -> IRId:
        i = rid
        if i in self.memo:
            return self.memo[i]

        n = self.src.nodes[i]

        match n:
            case TemporalUnary(kind=UnaryIROpKind.FINALLY, arg=arg, interval=iv, span=span):
                # rewrite: F φ  ==>  (True) U φ
                arg_id = self.visit(arg)
                t_id = self.dst.const_bool(True, span=span)
                out = self.dst.temporal_binary(BinaryIROpKind.UNTIL, t_id, arg_id, iv, span=span)

            case _:
                # Default cloning for everything else
                out = super().visit(rid)

        self.memo[i] = out
        return out


# ============================================================
# Pass 2: Combine GLOBALLY under AND into segmented Gs
#  - Emits explicit segments: [-inf, minB], finite [s,e-1], [maxB, +inf]
#  - Untimed G acts on every segment
# ============================================================


class PassCombineGloballySegments(IRRewriter):
    def visit(self, rid: IRId) -> IRId:
        i = rid
        if i in self.memo:
            return self.memo[i]
        n = self.src.nodes[i]

        match n:
            case Nary(kind=NaryKind.AND, args=args, span=and_span):
                # Recurse first
                kids = [self.visit(a) for a in args]

                # Collect sibling GLOBALLY nodes
                g_indices: list[int] = []
                g_nodes: list[TemporalUnary] = []
                for idx, kid in enumerate(kids):
                    kn = self.dst.nodes[kid]
                    if isinstance(kn, TemporalUnary) and kn.kind == UnaryIROpKind.GLOBALLY:
                        g_indices.append(idx)
                        g_nodes.append(kn)

                if len(g_nodes) < 2:
                    out = self.dst.nary(NaryKind.AND, kids, span=and_span)
                    self.memo[i] = out
                    return out

                # Partition timed vs untimed and collect boundaries
                bounds: list[int] = []
                timed: list[tuple[int, int, TemporalUnary]] = []
                untimed: list[TemporalUnary] = []
                for g in g_nodes:
                    iv = g.interval
                    if iv is not None and iv.lo is not None and iv.hi is not None:
                        s, e = iv.lo, iv.hi + 1
                        bounds.extend((s, e))
                        timed.append((s, e, g))
                    else:
                        untimed.append(g)
                bounds = sorted(set(bounds))

                # --- NEW BEHAVIOR: all G are untimed -> merge into single untimed G(AND(...)) ---
                if not bounds and len(untimed) == len(g_nodes):
                    # Build AND of all operands of the untimed Gs
                    ops = [u.arg for u in untimed]  # IRIds in dst already
                    # Join spans of operands for a nice span on the synthesized nodes
                    op_spans = [self.dst.nodes[o].span for o in ops]
                    conj_span = join_spans(op_spans, fallback=and_span)
                    conj = self.dst.nary(NaryKind.AND, ops, span=conj_span)

                    merged_g = self.dst.temporal_unary(UnaryIROpKind.GLOBALLY, conj, None, span=conj_span)

                    # Rebuild AND: put merged G at the position of the first G, drop originals
                    first = g_indices[0]
                    rebuilt: list[IRId] = []
                    for idx, kid in enumerate(kids):
                        if idx == first:
                            rebuilt.append(merged_g)
                        if idx in g_indices:
                            continue
                        rebuilt.append(kid)

                    new_span = join_spans([self.dst.nodes[x].span for x in rebuilt], fallback=and_span)
                    out = self.dst.nary(NaryKind.AND, rebuilt, span=new_span)
                    self.memo[i] = out
                    return out

                # ---- existing timed/mixed-segmentation logic (unchanged) ----
                if not bounds:
                    out = self.dst.nary(NaryKind.AND, kids, span=and_span)
                    self.memo[i] = out
                    return out

                minB, maxB = bounds[0], bounds[-1]
                untimed_ops = [g.arg for g in untimed]

                segments: list[IRId] = []

                def emit(lo: Optional[int], hi: Optional[int], ops: list[IRId]) -> None:
                    if not ops:
                        return
                    seg_span = join_spans([self.dst.nodes[o].span for o in ops], fallback=and_span)
                    conj = self.dst.nary(NaryKind.AND, ops, span=seg_span)
                    iv = None if (lo is None and hi is None) else IntervalIR(lo=lo, hi=hi)
                    segments.append(self.dst.temporal_unary(UnaryIROpKind.GLOBALLY, conj, iv, span=seg_span))

                # [-inf, minB]
                if untimed_ops:
                    emit(None, minB, untimed_ops)

                # Finite segments [s,e) → [s,e-1]
                for j in range(len(bounds) - 1):
                    s, e = bounds[j], bounds[j + 1]
                    if e <= s:
                        continue
                    active: list[IRId] = []
                    for ts, te, g in timed:
                        if ts <= s and te >= e:
                            active.append(g.arg)
                    active.extend(untimed_ops)
                    if active:
                        emit(s, e - 1, active)

                # [maxB, +inf]
                if untimed_ops:
                    emit(maxB, None, untimed_ops)

                # Rebuild AND with segments inserted at first G position
                first = g_indices[0]
                rebuilt: list[IRId] = []
                for idx, kid in enumerate(kids):
                    if idx == first:
                        rebuilt.extend(segments)
                    if idx in g_indices:
                        continue
                    rebuilt.append(kid)

                new_span = join_spans([self.dst.nodes[x].span for x in rebuilt], fallback=and_span)
                out = self.dst.nary(NaryKind.AND, rebuilt, span=new_span)

            case _:
                out = super().visit(rid)

        self.memo[i] = out
        return out


class DNFConverter(IRRewriter):
    def visit(self, rid: IRId) -> IRId:
        # Pass dnf through all temporal operators:
        # dnf( ( ... ) U ( ... ) ) = ( dnf( ... ) ) U ( dnf( ... ) )
        # dnf( G( ... ) ) = G( dnf( ... ) )
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
                raise ValueError("Binary nodes should have been converted to Nary... ?")

            case Nary(kind=kind, args=args, span=span):
                arg_ids = [self.visit(a) for a in args]

                if kind == NaryKind.OR:
                    ...
                elif kind == NaryKind.AND:
                    # Distribute OR over AND.
                    ...
                else:
                    raise ValueError(f"Unexpected Nary kind: {kind}")

                # out = self.rebuild_Nary(kind, arg_ids, span)

            case _:
                out = self.generic_visit(i, n)

        self.memo[i] = out
        return out
