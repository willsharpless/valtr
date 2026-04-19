from __future__ import annotations

from collections.abc import Iterable

import graphviz

from .lexer import Position, Span
from .ltl_ir import And, Const, ExistsPaths, Expr, ExprId, Finally, ForAllPaths, Globally, Interval, Next, Not, Or, Release, Until, Var


def _pos_str(pos: Position) -> str:
    return f"{pos.lineno}:{pos.col} (#{pos.index})"


def _span_str(span: Span) -> str:
    return f"{_pos_str(span.start)} .. {_pos_str(span.end)}"


def _interval_str(interval: Interval | None) -> str:
    if interval is None:
        return ""
    lo = "-inf" if interval.lo is None else str(interval.lo)
    hi = "inf" if interval.hi is None else str(interval.hi)
    return f"[{lo},{hi}]"


def _expr_title(node: Expr) -> str:
    if isinstance(node, Var):
        return f"Var {node.name}"
    if isinstance(node, Const):
        return f"Const {node.value}"
    if isinstance(node, Not):
        return "Not"
    if isinstance(node, And):
        return f"And ({len(node.args)})"
    if isinstance(node, Or):
        return f"Or ({len(node.args)})"
    if isinstance(node, Next):
        return f"Next {_interval_str(node.interval)}".rstrip()
    if isinstance(node, ForAllPaths):
        return "ForAllPaths"
    if isinstance(node, ExistsPaths):
        return "ExistsPaths"
    if isinstance(node, Finally):
        return f"Finally {_interval_str(node.interval)}".rstrip()
    if isinstance(node, Globally):
        return f"Globally {_interval_str(node.interval)}".rstrip()
    if isinstance(node, Until):
        return f"Until {_interval_str(node.interval)}".rstrip()
    if isinstance(node, Release):
        return f"Release {_interval_str(node.interval)}".rstrip()
    return type(node).__name__


def _origin_lines(node: Expr) -> list[str]:
    lines: list[str] = []
    origin = node.origin
    if origin.primary_span is not None:
        lines.append(_span_str(origin.primary_span))
    if origin.rule is not None:
        lines.append(f"rule={origin.rule}")
    if origin.sources:
        source_summary = ", ".join(
            f"{source.phase}:{source.node_id}" for source in sorted(origin.sources, key=lambda item: (item.phase, item.node_id))
        )
        lines.append(f"src={source_summary}")
    return lines


def _node_label(node_id: int, node: Expr) -> str:
    lines = [f"%{node_id}", _expr_title(node)]
    lines.extend(_origin_lines(node))
    return "\n".join(lines)


def _node_style(node: Expr) -> tuple[str, str]:
    if isinstance(node, (Var, Const)):
        return "#2E86AB", "ellipse"
    if isinstance(node, Not):
        return "#E67E22", "box"
    if isinstance(node, (And, Or)):
        return "#16A085", "box3d"
    if isinstance(node, (Next, ForAllPaths, ExistsPaths, Finally, Globally)):
        return "#F39C12", "box"
    if isinstance(node, (Until, Release)):
        return "#8E44AD", "diamond"
    return "#7F8C8D", "box"


def _normalize_roots(roots: Iterable[ExprId] | ExprId) -> list[int]:
    if isinstance(roots, ExprId):
        return [int(roots)]
    return [int(root) for root in roots]


def visualize_ltl_ir(
    nodes: list[Expr],
    roots: Iterable[ExprId] | ExprId,
    filename: str | None = None,
    view: bool = False,
    graph_name: str = "LTLIR",
    rankdir: str = "TB",
) -> graphviz.Digraph:
    """Create a Graphviz view of the LTL IR reachable from ``roots``."""

    root_ids = _normalize_roots(roots)

    seen: set[int] = set()
    stack = list(root_ids)
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(int(child) for child in nodes[node_id].children())

    dot = graphviz.Digraph(name=graph_name, graph_attr={"rankdir": rankdir, "splines": "true"})
    dot.attr("node", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    for node_id in sorted(seen):
        node = nodes[node_id]
        fillcolor, shape = _node_style(node)
        dot.node(
            f"n{node_id}",
            label=_node_label(node_id, node),
            shape=shape,
            style="filled,rounded",
            fillcolor=fillcolor,
            color="#34495E",
        )

    for node_id in sorted(seen):
        node = nodes[node_id]
        if isinstance(node, Not):
            dot.edge(f"n{node_id}", f"n{int(node.arg)}", label="arg")
        elif isinstance(node, (Next, ForAllPaths, ExistsPaths, Finally, Globally)):
            dot.edge(f"n{node_id}", f"n{int(node.arg)}", label="arg")
        elif isinstance(node, (Until, Release)):
            left = int(node.left)
            right = int(node.right)
            dot.edge(f"n{node_id}", f"n{left}", label="L")
            dot.edge(f"n{node_id}", f"n{right}", label="R")
            dot.edge(f"n{left}", f"n{right}", style="invis", weight="100")
        elif isinstance(node, (And, Or)):
            for index, child in enumerate(node.args):
                dot.edge(f"n{node_id}", f"n{int(child)}", label=str(index))

    for root_id in root_ids:
        if root_id in seen:
            dot.node(
                f"n{root_id}",
                label=_node_label(root_id, nodes[root_id]),
                penwidth="2.5",
                color="#C0392B",
            )

    if filename:
        dot.render(filename, view=view, cleanup=True)
    return dot
