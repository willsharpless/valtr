from attrs import frozen

from valtr.reachability import (
    DAGAvoid,
    DAGConst,
    DAGGUMinN,
    DAGGUSingle,
    DAGMaxN,
    DAGMinGuard,
    DAGMinN,
    DAGNegate,
    DAGNode,
    DAGReach,
    DAGReachAvoid,
    DAGVar,
)

MERMAID_EDGE_COLOR = "#2c3e50"
MERMAID_EDGE_STROKE_WIDTH = "2.25px"
MERMAID_NODE_PADDING = 8
MERMAID_DIAGRAM_PADDING = 48


@frozen
class DAGVizStyle:
    key: str
    graphviz_fill: str
    graphviz_shape: str
    terminal_truecolor: str
    terminal_256: int
    mermaid_fill: str
    mermaid_shape: str = "rounded"
    mermaid_text: str = "#111111"
    mermaid_stroke: str = "#2c3e50"
    mermaid_font_size: str = "25px"
    mermaid_stroke_width: str = "5px"
    mermaid_edge_label_fill: str | None = None
    mermaid_edge_label_text: str | None = None
    mermaid_edge_label_border: str | None = None


DEFAULT_DAG_STYLE = DAGVizStyle(
    key="default",
    graphviz_fill="#B98EC8",
    graphviz_shape="box",
    terminal_truecolor="#B98EC8",
    terminal_256=140,
    mermaid_fill="#B98EC8",
    mermaid_shape="rounded",
)


STYLE_BY_KEY = {
    "const": DAGVizStyle("const", "#95a5a6", "ellipse", "#95a5a6", 245, "#95a5a6", mermaid_shape="rounded"),
    "var": DAGVizStyle(
        "var",
        "#FFFFFF",
        "ellipse",
        "#FFFFFF",
        15,
        "#FFFFFF",
        mermaid_shape="rectangle",
        mermaid_font_size="20px",
        mermaid_stroke_width="5px",
    ),
    "min": DAGVizStyle(
        "min",
        "#E9E0C4",
        "box",
        "#E9E0C4",
        187,
        "#E9E0C4",
        mermaid_text="#FFFFFF",
        mermaid_shape="rounded",
        mermaid_font_size="15px",
        mermaid_stroke_width="5px",
    ),
    "max": DAGVizStyle(
        "max",
        "#D8CCA3",
        "box",
        "#D8CCA3",
        223,
        "#D8CCA3",
        mermaid_text="#FFFFFF",
        mermaid_shape="rounded",
        mermaid_font_size="15px",
        mermaid_stroke_width="5px",
    ),
    "reachavoid": DAGVizStyle(
        "reachavoid",
        "#1B5B93",
        "diamond",
        "#1B5B93",
        25,
        "#1B5B93",
        mermaid_shape="rounded",
        mermaid_text="#FFFFFF",
        mermaid_edge_label_fill="#DBE8F4",
        mermaid_edge_label_text="#1B5B93",
        mermaid_edge_label_border="#86AECB",
    ),
    "avoid": DAGVizStyle(
        "avoid",
        "#CD3A3A",
        "hexagon",
        "#CD3A3A",
        160,
        "#CD3A3A",
        mermaid_shape="rounded",
        mermaid_text="#FFFFFF",
        mermaid_edge_label_fill="#F7DDDD",
        mermaid_edge_label_text="#B23838",
        mermaid_edge_label_border="#D99898",
    ),
    "reach": DAGVizStyle(
        "reach",
        "#1B5B93",
        "diamond",
        "#1B5B93",
        25,
        "#1B5B93",
        mermaid_shape="rounded",
        mermaid_text="#FFFFFF",
        mermaid_edge_label_fill="#DBE8F4",
        mermaid_edge_label_text="#1B5B93",
        mermaid_edge_label_border="#86AECB",
    ),
    "gu": DAGVizStyle("gu", "#3AA655", "octagon", "#3AA655", 71, "#3AA655", mermaid_shape="rounded"),
    "gumin": DAGVizStyle("gumin", "#3AA655", "box", "#3AA655", 71, "#3AA655", mermaid_shape="rounded"),
    "negate": DAGVizStyle(
        "negate",
        "#B98EC8",
        "box",
        "#B98EC8",
        140,
        "#B98EC8",
        mermaid_text="#FFFFFF",
        mermaid_shape="rounded",
        mermaid_font_size="15px",
        mermaid_stroke_width="5px",
    ),
}


def node_style_key(node: DAGNode) -> str:
    match node:
        case DAGConst():
            return "const"
        case DAGVar():
            return "var"
        case DAGMinN() | DAGMinGuard():
            return "min"
        case DAGMaxN():
            return "max"
        case DAGReachAvoid():
            return "reachavoid"
        case DAGAvoid():
            return "avoid"
        case DAGReach():
            return "reach"
        case DAGGUMinN():
            return "gumin"
        case DAGGUSingle():
            return "gu"
        case DAGNegate():
            return "negate"
        case _:
            return DEFAULT_DAG_STYLE.key


def node_style(node: DAGNode) -> DAGVizStyle:
    return STYLE_BY_KEY.get(node_style_key(node), DEFAULT_DAG_STYLE)


# mermaid katex rendering is weak generally (fails in github)
# def node_label_short(node: DAGNode) -> str:
#     match node:
#         case DAGConst(value=value):
#             return f"Const {value}"
#         case DAGVar(name=name):
#             return f"Var '{name}'"
#         case DAGNegate():
#             return "$$-1$$"
#         case DAGMinN() | DAGMinGuard():
#             return "min"
#         case DAGMaxN():
#             return "max"
#         case DAGReachAvoid():
#             return "$$V_{RA}$$"
#         case DAGReach():
#             return "$$V_{R}$$"
#         case DAGGUSingle():
#             return "$$V_{{RA}_{\ell}}$$"
#         case DAGGUMinN():
#             return "min($$V_{{RA}_{\ell}}$$)"
#         case DAGAvoid():
#             return "$$V_{A}$$"
#         case _:
#             return type(node).__name__

def node_label_short(node: DAGNode) -> str:
    match node:
        case DAGConst(value=value):
            return f"Const {value}"
        case DAGVar(name=name):
            return f"Var '{name}'"
        case DAGNegate():
            return "neg."
        case DAGMinN() | DAGMinGuard():
            return "min"
        case DAGMaxN():
            return "max"
        case DAGReachAvoid():
            return "reach-avoid Value"
        case DAGReach():
            return "reach Value"
        case DAGGUSingle():
            return "reach-avoid-loop Value"
        case DAGGUMinN():
            return "min(reach-avoid-loop Value)"
        case DAGAvoid():
            return "avoid Value"
        case _:
            return type(node).__name__


def node_summary(node: DAGNode) -> str:
    match node:
        case DAGConst(value=value):
            return f"Const({value})"
        case DAGVar(name=name):
            return f"Var({name})"
        case DAGNegate(arg=arg):
            return f"Negate(%{arg})"
        case DAGMinN(args=args):
            return "Min(" + ", ".join(f"%{arg}" for arg in args) + ")"
        case DAGMinGuard(temporal_arg=temporal_arg, nontemporal_arg=nontemporal_arg):
            return f"MinGuard(%{temporal_arg}, %{nontemporal_arg})"
        case DAGMaxN(args=args):
            return "Max(" + ", ".join(f"%{arg}" for arg in args) + ")"
        case DAGReachAvoid(reach=reach, avoid=avoid):
            return f"ReachAvoid(reach=%{reach}, avoid=%{avoid})"
        case DAGReach(reach=reach):
            return f"Reach(%{reach})"
        case DAGAvoid(avoid=avoid):
            return f"Avoid(%{avoid})"
        case DAGGUSingle(reach=reach, avoid=avoid):
            return f"GU(reach=%{reach}, avoid=%{avoid})"
        case DAGGUMinN(args=args):
            return "Min(GU " + ", ".join(f"%{arg}" for arg in args) + ")"
        case _:
            return type(node).__name__


def with_alpha(hex_color: str, alpha_hex: str = "88") -> str:
    return f"{hex_color}{alpha_hex}"
