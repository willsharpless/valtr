from typing import Iterable, Set

from valtr.dag_viz_style import (
    MERMAID_DIAGRAM_PADDING,
    MERMAID_EDGE_COLOR,
    MERMAID_EDGE_STROKE_WIDTH,
    MERMAID_NODE_PADDING,
    STYLE_BY_KEY,
    node_label_short,
    node_style,
    node_style_key,
    with_alpha,
)
from valtr.reachability import DAGGUSingle, DAGReachAvoid, DagBuilder


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


def _escape_label(text: str) -> str:
    return text.replace('"', "&quot;")


def _with_alpha_hex(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    alpha_byte = max(0, min(255, round(alpha * 255)))
    return f"#{value}{alpha_byte:02X}"


def _edge_label_html(text: str, *, fill: str, border: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:2px 6px;border-radius:999px;'
        f'background:{_with_alpha_hex(fill, 0.96)};border:1px solid {_with_alpha_hex(border, 0.96)};'
        f'color:{color} !important;opacity:1;">{text}</span>'
    )


def _mermaid_node_label(i: int, node) -> str:
    short = node_label_short(node)
    style_key = node_style_key(node)
    if style_key == "var":
        return _escape_label(short.replace("Var ", ""))
    if style_key in {"min", "max", "gumin"}:
        return _escape_label(short)
    if style_key in {"negate"}:
        return _escape_label(f"{short}")
    if style_key in {"reachavoid", "reach", "gu", "avoid"}:
        # return _escape_label(f"n.{i}: Value <br/>{short} <br/> </br> hello")
        # return _escape_label(f"n.{i}: <br/> Value {short}")
        # return _escape_label(f"n.{i}{short}")
        return _escape_label(f"{short} (n.{i})")


def render_dag_mermaid(
    builder: DagBuilder,
    roots: Iterable[int] | int,
    direction: str = "TD",
    hide_avoid: bool = False,
) -> str:
    if isinstance(roots, int):
        root_ids = [roots]
    else:
        root_ids = list(roots)
    seen = _reachable(builder, root_ids)

    lines = [
        f'%%{{init: {{"theme":"base","htmlLabels":true,"flowchart":{{"padding":{MERMAID_NODE_PADDING},"diagramPadding":{MERMAID_DIAGRAM_PADDING}}},"themeVariables":{{"background":"#FFFFFF00","fontFamily":"JetBrains Mono, Roboto Mono, Menlo, Consolas, monospace","edgeLabelBackground":"#FFFFFF","lineColor":"{MERMAID_EDGE_COLOR}","defaultLinkColor":"{MERMAID_EDGE_COLOR}"}},"themeCSS":".edgeLabel rect {{ fill: #FFFFFF !important; stroke: {MERMAID_EDGE_COLOR} !important; stroke-width: 1px !important; rx: 999px; ry: 999px; }} .edgeLabel span, .edgeLabel p {{ background: transparent !important; }} .edgeLabel foreignObject {{ background: transparent !important; }} .labelBkg {{ fill: #FFFFFF !important; }} .flowchart-link, .edgePath path, .edge-thickness-normal, .edge-thickness-thick {{ stroke: {MERMAID_EDGE_COLOR} !important; stroke-width: {MERMAID_EDGE_STROKE_WIDTH} !important; fill: none !important; }} .arrowheadPath {{ stroke: {MERMAID_EDGE_COLOR} !important; stroke-width: {MERMAID_EDGE_STROKE_WIDTH} !important; fill: {MERMAID_EDGE_COLOR} !important; }} svg {{ background-color: transparent; }}"}}%%',
        f"flowchart {direction}",
    ]

    for i in sorted(seen):
        node = builder.nodes[i]
        label = _mermaid_node_label(i, node)
        shape = node_style(node).mermaid_shape
        lines.append(f'    n{i}@{{ shape: {shape}, label: "{label}" }}')

    for i in sorted(seen):
        node = builder.nodes[i]
        if isinstance(node, (DAGReachAvoid, DAGGUSingle)):
            if node.reach in seen:
                reach_label = _edge_label_html(
                    "reach",
                    fill="#FFFFFF",
                    border="#86AECB",
                    color="#4698E0",
                )
                lines.append(f"    n{i} ==>|{reach_label}| n{node.reach}")
            if not hide_avoid and node.avoid in seen:
                avoid_label = _edge_label_html(
                    "avoid",
                    fill="#F7DDDD",
                    border="#D99898",
                    color="#D95454",
                )
                lines.append(f"    n{i} ==>|{avoid_label}| n{node.avoid}")
            continue

        for child in node.children():
            if child in seen:
                lines.append(f"    n{i} ==> n{child}")

    for i in sorted(seen):
        style_key = node_style_key(builder.nodes[i])
        lines.append(f"    class n{i} {style_key};")

    for style_key, style in STYLE_BY_KEY.items():
        lines.append(
            f"    classDef {style_key} fill:{with_alpha(style.mermaid_fill)},stroke:{style.mermaid_fill},color:{style.mermaid_text},stroke-width:{style.mermaid_stroke_width},font-size:{style.mermaid_font_size};"
        )

    for root in root_ids:
        if root in seen:
            root_style = node_style(builder.nodes[root])
            lines.append(f"    style n{root} stroke:{root_style.mermaid_fill},stroke-width:8px;")

    return "\n".join(lines) + "\n"
