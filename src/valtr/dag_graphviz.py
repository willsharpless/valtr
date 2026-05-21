from typing import Iterable, List, Optional, Set

import graphviz

from valtr.dag_viz_style import DEFAULT_DAG_STYLE, node_label_short, node_style
from valtr.reachability import (DAGAvoid, DagBuilder, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN,
                                DAGMinGuard, DAGNegate, DAGNode, DAGReachAvoid, DAGVar)


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
    return f"n.{i}\n{node_label_short(node)}"


def visualize_dag(
    builder: DagBuilder,
    roots: Iterable[int] | int,
    filename: Optional[str] = None,
    view: bool = False,
    graph_name: str = "DAG",
    rankdir: str = "TB",  # or "LR"
    hide_avoid: bool = False
) -> graphviz.Digraph:
    """
    Render the reachable DAG as a Graphviz Digraph.

    - Left/right ordering is enforced for RA(left, right).
    - Multiple roots are supported and highlighted.
    if hide_avoid is True, then don't render the "avoid" edges in DAGReachAvoid and DAGGUSingle, to reduce clutter.
    """
    # Collect reachable nodes
    if isinstance(roots, int):
        root_ids = [roots]
    else:
        root_ids = list(roots)
    seen = _reachable(builder, root_ids)

    dot = graphviz.Digraph(name=graph_name, graph_attr={"rankdir": rankdir, "splines": "true"})
    dot.attr("node", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # Nodes
    for i in sorted(seen):
        node = builder.nodes[i]
        style = node_style(node)
        color = style.graphviz_fill or DEFAULT_DAG_STYLE.graphviz_fill
        shape = style.graphviz_shape or DEFAULT_DAG_STYLE.graphviz_shape
        dot.node(
            f"n{i}", label=_label_for(i, node), shape=shape, style="filled,rounded", fillcolor=color, color="#2c3e50"
        )

    # Edges (with ordering for RA)
    for i in sorted(seen):
        node = builder.nodes[i]

        if isinstance(node, DAGReachAvoid) or isinstance(node, DAGGUSingle):
            l, r = node.reach, node.avoid
            if l in seen:
                dot.edge(f"n{i}", f"n{l}", label="Reach")

            if not hide_avoid:
                if r in seen:
                    dot.edge(f"n{i}", f"n{r}", label="Avoid")
                # Enforce left→right order visually
                if l in seen and r in seen:
                    dot.edge(f"n{l}", f"n{r}", style="invis", weight="100", constraint="true")
            continue
        # elif isinstance(node, DAG):
        #     for q, r in node.args:
        #         # Create an "until" node for each (q,r) pair.
        #         until_node_name = f"n{q}_{r}"
        #         until_color, until_shape = color_map[DAGReachAvoid]
        #         dot.node(
        #             until_node_name,
        #             label="ReachAvoid",
        #             shape=until_shape,
        #             style="filled,rounded",
        #             fillcolor=until_color,
        #             color="#2c3e50",
        #         )
        #         dot.edge(f"n{i}", until_node_name)
        #
        #         # Add the q and r below the above node.
        #         dot.edge(until_node_name, f"n{q}", label="q")
        #         dot.edge(until_node_name, f"n{r}", label="r")
        #
        #         # Enforce left->right order visually
        #         dot.edge(f"n{q}", f"n{r}", style="invis", weight="100", constraint="true")
        #     continue

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
