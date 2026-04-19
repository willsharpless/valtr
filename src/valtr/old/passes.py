from typing import Iterable, List, Tuple

import ipdb

from ..lexer import Span
from ..tl_parser import (BinaryOp, BinaryOpKind, Identifier, Interval, MultiFinally, NaryBool, Node, UnaryOp,
                             UnaryOpKind)


def _join_span(a: Span, b: Span) -> Span:
    return Span(start=a.start, end=b.end)


def _join_spans_of_nodes(nodes: list[Node]) -> Span:
    assert nodes, "Cannot join spans of empty list"
    return _join_span(nodes[0].span, nodes[-1].span)


def _make_and(children: list[Node]) -> Node:
    """Return an n-ary AND (NaryBool) or the only child if len == 1."""
    assert children, "AND needs at least one child"
    if len(children) == 1:
        return children[0]
    return NaryBool(kind=BinaryOpKind.AND, children=children, span=_join_spans_of_nodes(children))


def _mk_interval(lo: int, hi: int, span_hint: Span) -> Interval:
    return Interval(lo=lo, hi=hi, span=span_hint)


def _mk_globally(operand: Node, interval: Interval | None, span_hint: Span) -> UnaryOp:
    # span: cover operator-ish region and operand; we use span_hint ∪ operand.span
    full_span = _join_span(span_hint, operand.span)
    return UnaryOp(op=UnaryOpKind.GLOBALLY, operand=operand, interval=interval, span=full_span)


def _flatten_nary(kind: BinaryOpKind, nodes: Iterable[Node]) -> list[Node]:
    """
    Given nodes that are already optimized, collect children:
    - If a child is NaryBool of the same kind, inline its children.
    - If a child is BinaryOp of the same kind, convert it (and inline its 2 children).
    - Otherwise, keep the child.
    """
    out: list[Node] = []
    for n in nodes:
        if isinstance(n, NaryBool) and n.kind == kind:
            out.extend(n.children)
        elif isinstance(n, BinaryOp) and n.op == kind:
            # Shouldn't happen if we always convert; still, be safe and explode it.
            out.extend([n.left, n.right])
        else:
            out.append(n)
    return out


def normalize_nary_bool(node: Node) -> Node:
    """
    Transform the AST so:
      - All AND/OR subtrees are represented by NaryBool
      - Those NaryBool nodes are fully flattened (no nested same-kind)
    """
    # 1) Recurse first
    if isinstance(node, Identifier) or isinstance(node, Interval):
        return node

    if isinstance(node, UnaryOp):
        operand = normalize_nary_bool(node.operand)
        if operand is node.operand:
            return node
        return UnaryOp(op=node.op, operand=operand, interval=node.interval, span=_join_span(node.span, operand.span))

    if isinstance(node, BinaryOp):
        left = normalize_nary_bool(node.left)
        right = normalize_nary_bool(node.right)

        if node.op in (BinaryOpKind.AND, BinaryOpKind.OR):
            kind = node.op
            # Convert this binary into an N-ary, then flatten any nested same-kinds
            children = _flatten_nary(kind, [left, right])
            # Keep flattening if grandchildren still contain same-kind BinaryOp/NaryBool
            changed = True
            while changed:
                changed = False
                new_children = []
                for c in children:
                    if isinstance(c, NaryBool) and c.kind == kind:
                        new_children.extend(c.children)
                        changed = True
                    elif isinstance(c, BinaryOp) and c.op == kind:
                        new_children.extend([c.left, c.right])
                        changed = True
                    else:
                        new_children.append(c)
                children = new_children

            if len(children) == 1:
                # Degenerate case (rare): AND/OR with a single child — return the child
                return children[0]

            span = _join_span(children[0].span, children[-1].span)
            return NaryBool(kind=kind, children=children, span=span)

        # Non-boolean binaries (IMPLIES / UNTIL / RELEASE) remain binary
        return BinaryOp(
            op=node.op, left=left, right=right, interval=node.interval, span=_join_span(left.span, right.span)
        )

    if isinstance(node, NaryBool):
        # Recurse into children, then flatten again to guarantee invariants.
        kids = [normalize_nary_bool(c) for c in node.children]
        kids = _flatten_nary(node.kind, kids)
        if len(kids) == 1:
            return kids[0]
        span = _join_span(kids[0].span, kids[-1].span)
        return NaryBool(kind=node.kind, children=kids, span=span)

    # Fallback (unknown node type)
    return node


def combine_multi_finally(node: Node) -> Node:
    """Converts all conjunctions of FINALLY operators into a single MultiFinally node.

    The transformation is local to each NaryBool(AND):
        AND( F a, F_[1,2] b, c, F d )  ==>  AND( MultiFinally([a,b,d],[None,[1,2],None]), c )
    """
    # Recurse
    if isinstance(node, Identifier):
        return node

    if isinstance(node, UnaryOp):
        operand_new = combine_multi_finally(node.operand)
        if operand_new is node.operand:
            return node
        return UnaryOp(
            op=node.op, operand=operand_new, interval=node.interval, span=_join_span(node.span, operand_new.span)
        )

    if isinstance(node, BinaryOp):
        left_new = combine_multi_finally(node.left)
        right_new = combine_multi_finally(node.right)
        if left_new is node.left and right_new is node.right:
            return node
        return BinaryOp(
            op=node.op,
            left=left_new,
            right=right_new,
            interval=node.interval,
            span=_join_span(left_new.span, right_new.span),
        )

    if isinstance(node, NaryBool):
        # Recurse into children
        new_children = [combine_multi_finally(ch) for ch in node.children]

        if node.kind != BinaryOpKind.AND:
            # Non-AND NaryBool: leave as-is (only children were updated)
            return NaryBool(kind=node.kind, children=new_children, span=_join_spans_of_nodes(new_children))

        # Partition FINALLY vs non-FINALLY among children
        finally_indices: list[int] = []
        finally_ops: list[UnaryOp] = []
        for idx, ch in enumerate(new_children):
            if isinstance(ch, UnaryOp) and ch.op == UnaryOpKind.FINALLY:
                finally_indices.append(idx)
                finally_ops.append(ch)

        # If there are at least two FINALLY nodes, combine them
        if len(finally_ops) >= 2:
            # Build MultiFinally: collect operands/intervals in appearance order
            mf_operands: list[Node] = []
            mf_intervals: list[Interval | None] = []
            for f in finally_ops:
                mf_operands.append(f.operand)
                mf_intervals.append(f.interval)

            # Span of MultiFinally: from first FINALLY to last FINALLY
            mf_span = _join_span(finally_ops[0].span, finally_ops[-1].span)
            multi = MultiFinally(operands=mf_operands, intervals=mf_intervals, span=mf_span)

            # Build the new child list:
            # - Keep non-FINALLYs
            # - Remove FINALLYs
            # - Insert MultiFinally at the position of the first FINALLY encountered
            keep: list[Node] = []
            insert_pos = finally_indices[0]
            for i, ch in enumerate(new_children):
                if i in finally_indices:
                    continue
                keep.append(ch)

            # Insert MultiFinally at the appropriate place
            keep.insert(insert_pos, multi)

            # If len(keep) == 1, remove the AND.
            if len(keep) == 1:
                return keep[0]

            # Return the rebuilt AND node with updated span
            return NaryBool(kind=BinaryOpKind.AND, children=keep, span=_join_spans_of_nodes(keep))

        # Otherwise, leave as-is (just children updated)
        return NaryBool(kind=BinaryOpKind.AND, children=new_children, span=_join_spans_of_nodes(new_children))

    # Unknown node kind: return unchanged
    return node


def combine_multi_globally(root: Node) -> Node:
    """
    Under each NaryBool(AND, ...), merge sibling G nodes:

    * All untimed  -> one G over the n-ary AND of operands.
    * Any timed    -> build non-overlapping finite segments from timed boundaries
                      and also emit the two infinite segments:
                        [-inf, minB] and [maxB, +inf]
                      Untimed G participate in *every* segment (finite and infinite).
    """

    def walk(n: Node) -> Node:
        # Recurse bottom-up
        if isinstance(n, Identifier) or isinstance(n, Interval):
            return n

        if isinstance(n, UnaryOp):
            opnd = walk(n.operand)
            if opnd is n.operand:
                return n
            return UnaryOp(op=n.op, operand=opnd, interval=n.interval, span=_join_span(n.span, opnd.span))

        if isinstance(n, BinaryOp):
            l = walk(n.left)
            r = walk(n.right)
            if l is n.left and r is n.right:
                return n
            return BinaryOp(op=n.op, left=l, right=r, interval=n.interval, span=_join_span(l.span, r.span))

        if isinstance(n, NaryBool):
            kids = [walk(c) for c in n.children]

            if n.kind != BinaryOpKind.AND:
                return NaryBool(kind=n.kind, children=kids, span=_join_spans_of_nodes(kids))

            # Collect sibling G nodes
            g_idxs: List[int] = []
            g_nodes: List[UnaryOp] = []
            for i, ch in enumerate(kids):
                if isinstance(ch, UnaryOp) and ch.op == UnaryOpKind.GLOBALLY:
                    g_idxs.append(i)
                    g_nodes.append(ch)

            if len(g_nodes) < 2:
                return NaryBool(kind=BinaryOpKind.AND, children=kids, span=_join_spans_of_nodes(kids))

            all_untimed = all(g.interval is None for g in g_nodes)
            if all_untimed:
                conj = _make_and([g.operand for g in g_nodes])
                span_hint = _join_span(g_nodes[0].span, g_nodes[-1].span)
                merged = _mk_globally(conj, None, span_hint)

                # Rebuild
                first = g_idxs[0]
                rebuilt: List[Node] = []
                for i, ch in enumerate(kids):
                    if i == first:
                        rebuilt.append(merged)
                    if i in g_idxs:
                        continue
                    rebuilt.append(ch)
                return NaryBool(kind=BinaryOpKind.AND, children=rebuilt, span=_join_spans_of_nodes(rebuilt))

            # Timed or mixed case
            # 1) Gather finite boundaries from all *timed* G nodes
            bounds: List[int] = []
            half_open: List[Tuple[int, int, UnaryOp]] = []  # [lo, hi+1)
            for g in g_nodes:
                if g.interval is not None and g.interval.lo is not None and g.interval.hi is not None:
                    lo, hi = g.interval.lo, g.interval.hi
                    s, e = lo, hi + 1
                    bounds.append(s)
                    bounds.append(e)
                    half_open.append((s, e, g))
            bounds = sorted(set(bounds))

            # If for some reason there are no finite boundaries, fall back
            if not bounds:
                return NaryBool(kind=BinaryOpKind.AND, children=kids, span=_join_spans_of_nodes(kids))

            minB, maxB = bounds[0], bounds[-1]

            # 2) Separate untimed Gs (active everywhere)
            untimed_gs: List[UnaryOp] = [g for g in g_nodes if g.interval is None]
            untimed_ops: List[Node] = [g.operand for g in untimed_gs]
            untimed_spans: List[Span] = [g.span for g in untimed_gs]

            new_gs: List[UnaryOp] = []

            # 3) Infinite segment: [-inf, minB]
            if untimed_ops:
                conj = _make_and(list(untimed_ops))
                span_hint = (
                    untimed_spans[0] if len(untimed_spans) == 1 else _join_span(untimed_spans[0], untimed_spans[-1])
                )
                iv = _mk_interval(None, minB, span_hint)  # None => -inf
                new_gs.append(_mk_globally(conj, iv, span_hint))

            # 4) Finite segments [s, e) across all boundaries, back to inclusive [s, e-1]
            for i in range(len(bounds) - 1):
                s, e = bounds[i], bounds[i + 1]
                if e <= s:
                    continue

                active_ops: List[Node] = []
                active_spans: List[Span] = []

                # Timed actives that fully cover [s, e)
                for lo, hi1, g in half_open:
                    if lo <= s and hi1 >= e:
                        active_ops.append(g.operand)
                        active_spans.append(g.span)

                # Untimed actives (present on every segment)
                if untimed_ops:
                    active_ops.extend(untimed_ops)
                    active_spans.extend(untimed_spans)

                if not active_ops:
                    continue

                conj = _make_and(active_ops)
                iv_span_hint = (
                    active_spans[0] if len(active_spans) == 1 else _join_span(active_spans[0], active_spans[-1])
                )
                iv = _mk_interval(s, e - 1, iv_span_hint)
                new_gs.append(_mk_globally(conj, iv, iv_span_hint))

            # 5) Infinite segment: [maxB, +inf]
            if untimed_ops:
                conj = _make_and(list(untimed_ops))
                span_hint = (
                    untimed_spans[0] if len(untimed_spans) == 1 else _join_span(untimed_spans[0], untimed_spans[-1])
                )
                iv = _mk_interval(maxB, None, span_hint)  # None => +inf
                new_gs.append(_mk_globally(conj, iv, span_hint))

            # (Optional) coalesce adjacent finite segments with identical ops
            # Skipped here for clarity; re-add if desired.

            # Rebuild: place new Gs where first G was; remove original Gs
            first = g_idxs[0]
            rebuilt: List[Node] = []
            for i, ch in enumerate(kids):
                if i == first:
                    rebuilt.extend(new_gs)
                if i in g_idxs:
                    continue
                rebuilt.append(ch)

            return NaryBool(kind=BinaryOpKind.AND, children=rebuilt, span=_join_spans_of_nodes(rebuilt))

        return n

    return walk(root)
