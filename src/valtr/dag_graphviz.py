from typing import Iterable, List, Optional, Set

import graphviz

from valtr.reachability import (DAGAvoid, DagBuilder, DAGConst, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGReachAvoid, DAGVar)


def _reachable(builder: DagBuilder, roots: Iterable[int] | int) -> Set[int]:
    if isinstance(roots, int):
        stack = [roots]
    else:
        stack = list(roots)
    seen: Set[int] = set()
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        node = builder.nodes[i]
        stack.extend(node.children())
    return seen


def _label_for(i: int, node: DAGNode) -> str:
    # Multiline labels: ID on top, op/info below
    match node:
        case DAGConst(value=value):
            op = f"Const {value}"
        case DAGVar(name=name):
            op = f"Var {name}"
        case DAGNegate(arg=_):
            op = f"NEG."
        case DAGMinN(args=_):
            op = f"MIN"
        case DAGMaxN(args=_):
            op = f"MAX"
        case DAGReachAvoid(reach=_, avoid=_):
            op = "ReachAvoid"
        case DAGAvoid(avoid=_):
            op = "Avoid"
        case _:
            op = type(node).__name__

    return f"n.{i}\n{op}"


def visualize_dag(
    builder: DagBuilder,
    roots: Iterable[int] | int,
    filename: Optional[str] = None,
    view: bool = False,
    graph_name: str = "DAG",
    rankdir: str = "TB",  # or "LR"
) -> graphviz.Digraph:
    """
    Render the reachable DAG as a Graphviz Digraph.

    - Left/right ordering is enforced for RA(left, right).
    - Multiple roots are supported and highlighted.
    """
    # Collect reachable nodes
    if isinstance(roots, int):
        root_ids = [roots]
    else:
        root_ids = list(roots)
    seen = _reachable(builder, root_ids)

    # Styling
    color_map = {
        DAGConst: ("#95a5a6", "ellipse"),
        DAGVar: ("#FFFFFF", "ellipse"),
        DAGMinN: ("#61655D", "box"),
        DAGMaxN: ("#E9E0C4", "box"),
        DAGReachAvoid: ("#1B5B93", "diamond"),
        DAGAvoid: ("#CD3A3A", "hexagon"),
    }

    dot = graphviz.Digraph(name=graph_name, graph_attr={"rankdir": rankdir, "splines": "true"})
    dot.attr("node", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # Nodes
    for i in sorted(seen):
        node = builder.nodes[i]
        color, shape = ("#b98ec8", "box")
        for cls, (c, s) in color_map.items():
            if isinstance(node, cls):
                color, shape = c, s
                break
        dot.node(
            f"n{i}", label=_label_for(i, node), shape=shape, style="filled,rounded", fillcolor=color, color="#2c3e50"
        )

    # Edges (with ordering for RA)
    for i in sorted(seen):
        node = builder.nodes[i]

        if isinstance(node, DAGReachAvoid):
            l, r = node.reach, node.avoid
            if l in seen:
                dot.edge(f"n{i}", f"n{l}", label="Reach")
            if r in seen:
                dot.edge(f"n{i}", f"n{r}", label="Avoid")
            # Enforce left→right order visually
            if l in seen and r in seen:
                dot.edge(f"n{l}", f"n{r}", style="invis", weight="100", constraint="true")
            continue

        # Default edges for others
        for c in node.children():
            if c in seen:
                dot.edge(f"n{i}", f"n{c}")

    # Highlight roots
    for r in root_ids:
        if r in seen:
            # Thicker border on roots
            node = builder.nodes[r]
            dot.node(f"n{r}", label=_label_for(r, node), penwidth="2.5", color="#000000")

    if filename:
        dot.render(filename, view=view, cleanup=True)  # default PDF; set dot.format for PNG
    return dot
