from typing import Iterable, List, Optional, Set, Tuple

import graphviz

from valtr.ir import Binary, ConstBool, IRId, IRNode, Nary, TemporalBinary, TemporalUnary, Unary, Var
from valtr.ir_builder import IRBuilder
from valtr.lexer import Position, Span


def _pos_str(p: Position) -> str:
    # Position has lineno, col, index
    return f"{p.lineno}:{p.col} (#{p.index})"


def _span_str(span: Span) -> str:
    return f"{_pos_str(span.start)} .. {_pos_str(span.end)}"


def _iv_str(iv) -> str:
    if iv is None:
        return ""
    lo = "-inf" if iv.lo is None else str(iv.lo)
    hi = "inf" if iv.hi is None else str(iv.hi)
    return f"[{lo},{hi}]"


def _node_label(i: int, n: IRNode) -> str:
    match n:
        case Var(name=name, span=span):
            op = f"Var {name}"
        case ConstBool(value=val, span=span):
            op = f"Const {val}"
        case TemporalUnary(kind=k, interval=iv, span=span):
            op = f"{k.name} {_iv_str(iv)}"
        case Unary(kind=k, span=span):
            op = f"{k.name}"
        case TemporalBinary(kind=k, interval=iv, span=span):
            op = f"{k.name} {_iv_str(iv)}"
        case Binary(kind=k, span=span):
            op = f"{k.name}"
        case Nary(kind=k, args=args, span=span):
            op = f"{k.name} (n-ary)"
        case _:
            op = type(n).__name__

    return f"%{i}\n{op}\n{_span_str(n.span)}"


def _children_of(node) -> List[int]:
    if isinstance(node, TemporalUnary):
        return [int(node.arg)]
    if isinstance(node, Unary):
        return [int(node.arg)]
    if isinstance(node, TemporalBinary):
        return [int(node.left), int(node.right)]
    if isinstance(node, Binary):
        return [int(node.left), int(node.right)]
    if isinstance(node, Nary):
        return [int(a) for a in node.args]
    return []


# ---- main API ----


def visualize_ir(
    builder: IRBuilder,
    roots: Iterable[IRId] | IRId,
    filename: Optional[str] = None,
    view: bool = False,
    graph_name: str = "IR",
    rankdir: str = "TB",
) -> graphviz.Digraph:
    """
    Create a Graphviz Digraph for the IR reachable from `roots`.

    Args:
        builder: your IRBuilder instance (must have `nodes: list[IRNode]`)
        roots: a single IRId or an iterable of IRId roots to visualize
        filename: if provided, calls dot.render(filename, format='pdf' by default)
        view: if True and filename provided, opens the rendered file
        graph_name: Graphviz graph name
        rankdir: 'TB' (top-bottom) or 'LR' (left-right)

    Returns:
        graphviz.Digraph that you can further customize or render.
    """
    # normalize roots
    if isinstance(roots, IRId):
        root_ids = [int(roots)]
    else:
        root_ids = [int(r) for r in roots]

    # collect reachable
    seen: Set[int] = set()
    stack: List[int] = list(root_ids)
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        node = builder.nodes[i]
        stack.extend(_children_of(node))

    # styling
    color_map = {
        Var: ("#2E86AB", "ellipse"),
        ConstBool: ("#2E86AB", "ellipse"),
        Unary: ("#E67E22", "box"),
        TemporalUnary: ("#E67E22", "box"),
        Binary: ("#8E44AD", "diamond"),
        TemporalBinary: ("#8E44AD", "diamond"),
        Nary: ("#16A085", "box3d"),
    }

    dot = graphviz.Digraph(name=graph_name, graph_attr={"rankdir": rankdir, "splines": "true"})
    dot.attr("node", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # add nodes
    for i in sorted(seen):
        node = builder.nodes[i]
        # pick style by most specific class first
        color, shape = ("#7f8c8d", "box")
        for cls, (c, s) in color_map.items():
            if isinstance(node, cls):
                color, shape = c, s
                break

        label = _node_label(i, node)
        dot.node(f"n{i}", label=label, shape=shape, style="filled,rounded", fillcolor=color, color="#34495e")

    # Add edges with ordering constraints
    for i in sorted(seen):
        node = builder.nodes[i]
        kids = node.children()

        match node:
            case Binary(kind=kind, left=left, right=right, span=span):
                l = int(left)
                r = int(right)

                # Normal edges
                dot.edge(f"n{i}", f"n{l}", label="L", color="black")
                dot.edge(f"n{i}", f"n{r}", label="R", color="black")

                # Enforce horizontal ordering left → right
                dot.edge(f"n{l}", f"n{r}", style="invis", weight="100")  # not drawn!  # strong constraint to keep order

                # Group in same rank
                dot.edge(f"n{l}", f"n{r}", style="invis", constraint="true")

            case TemporalBinary(kind=kind, left=left, right=right, interval=iv, span=span):
                l = int(left)
                r = int(right)
                dot.edge(f"n{i}", f"n{l}", label="L")
                dot.edge(f"n{i}", f"n{r}", label="R")
                dot.edge(f"n{l}", f"n{r}", style="invis", weight="100")
                dot.edge(f"n{l}", f"n{r}", style="invis", constraint="true")

            case _:
                # default: just normal edges
                for c in kids:
                    j = int(c)
                    dot.edge(f"n{i}", f"n{j}")

    # mark roots
    for r in root_ids:
        if r in seen:
            dot.node(f"n{r}", _node_label(r, builder.nodes[r]), penwidth="2.5", color="#c0392b")

    if filename:
        # default render format PDF; you can pass dot.format='png' after calling
        dot.render(filename, view=view, cleanup=True)
    return dot
