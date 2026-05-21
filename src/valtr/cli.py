import argparse
import os
import pickle
import sys
from pathlib import Path

from valtr.dag_mermaid import render_dag_mermaid
from valtr.dag_viz_style import node_style, node_summary
from valtr.reachability import (
    DAGAvoid,
    DAGGUMinN,
    DAGGUSingle,
    DAGMinGuard,
    DAGNegate,
    DAGReach,
    DAGReachAvoid,
)
from valtr.valtr import to_dag


_RESET = "\033[0m"
_DIM = "\033[2m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _supports_truecolor() -> bool:
    return os.environ.get("COLORTERM", "").lower() == "truecolor"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _fg_truecolor(hex_color: str, text: str) -> str:
    r, g, b = _rgb(hex_color)
    return f"\033[38;2;{r};{g};{b}m{text}{_RESET}"


def _fg_256(code: int, text: str) -> str:
    return f"\033[38;5;{code}m{text}{_RESET}"


def _style(text: str, *, dim: bool = False) -> str:
    if not dim:
        return text
    return f"{_DIM}{text}{_RESET}"


def _ascii_lines(nodes: list, root: int, *, color: bool = False) -> list[str]:
    lines: list[str] = []
    seen: set[int] = set()
    use_truecolor = color and _supports_truecolor()

    def walk(node_id: int, prefix: str, edge_label: str | None, is_last: bool):
        node = nodes[node_id]
        branch = ""
        if prefix or edge_label is not None:
            branch = "└─ " if is_last else "├─ "
        label_prefix = f"{edge_label}: " if edge_label else ""
        summary = node_summary(node)
        node_text = f"%{node_id} {summary}"
        if color:
            style = node_style(node)
            if use_truecolor:
                node_text = _fg_truecolor(style.terminal_truecolor, node_text)
            else:
                node_text = _fg_256(style.terminal_256, node_text)
        branch_text = _style(branch, dim=True) if color else branch
        prefix_text = _style(prefix, dim=True) if color else prefix
        edge_text = _style(label_prefix, dim=True) if color else label_prefix
        shared_text = _style(" [shared]", dim=True) if color else " [shared]"

        if node_id in seen:
            lines.append(f"{prefix_text}{branch_text}{edge_text}{node_text}{shared_text}")
            return

        seen.add(node_id)
        lines.append(f"{prefix_text}{branch_text}{edge_text}{node_text}")

        children = list(node.children())
        next_prefix = prefix + ("   " if is_last else "│  ")
        for idx, child_id in enumerate(children):
            child_is_last = idx == len(children) - 1
            child_label = None
            match node:
                case DAGReachAvoid(reach=reach, avoid=avoid):
                    child_label = "reach" if child_id == reach else "avoid"
                case DAGGUSingle(reach=reach, avoid=avoid):
                    child_label = "reach" if child_id == reach else "avoid"
                case DAGReach(reach=_):
                    child_label = "reach"
                case DAGAvoid(avoid=_):
                    child_label = "avoid"
                case DAGNegate(arg=_):
                    child_label = "arg"
                case DAGMinGuard(temporal_arg=temporal_arg, nontemporal_arg=nontemporal_arg):
                    child_label = "temporal" if child_id == temporal_arg else "static"
            walk(child_id, next_prefix, child_label, child_is_last)

    walk(root, "", None, True)
    return lines


def _save_dag(path: Path, spec: str, dag, root: int):
    payload = {
        "spec": spec,
        "dag_nodes": dag.nodes,
        "dag_root": root,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _save_mermaid(path: Path, dag, root: int):
    text = render_dag_mermaid(dag, root)
    path.write_text(text)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="valtr",
        description="Parse a temporal-logic spec and build its value DAG.",
    )
    parser.add_argument("spec", help="Temporal-logic specification string.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render the DAG with Graphviz instead of printing the ASCII view.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Serialize the DAG to a pickle file in the current directory.",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Write Mermaid graph text to a .mmd file in the current directory.",
    )
    parser.add_argument(
        "--vertical",
        action="store_true",
        help="When used with --mermaid, render the Mermaid graph top-down instead of left-to-right.",
    )
    args = parser.parse_args(argv)

    dag, root = to_dag(args.spec)

    if not args.plot and not args.save and not args.mermaid:
        use_color = _supports_color()
        for line in _ascii_lines(dag.nodes, root, color=use_color):
            print(line)
        return

    if args.plot:
        try:
            from graphviz.backend.execute import ExecutableNotFound
            from valtr.dag_graphviz import visualize_dag
        except ModuleNotFoundError as exc:
            raise SystemExit("Plotting requires the Python 'graphviz' package to be installed.") from exc

        out_base = Path.cwd() / "value_tree_dag"
        try:
            dot = visualize_dag(dag, root, filename=str(out_base), view=False)
        except ExecutableNotFound as exc:
            raise SystemExit(
                "Plotting requires the Graphviz 'dot' executable to be installed and on PATH."
            ) from exc
        out_path = out_base.with_suffix(f".{dot.format or 'pdf'}")
        print(f"Wrote plot to {out_path}")

    if args.save:
        out_path = Path.cwd() / "value_tree_dag.pkl"
        _save_dag(out_path, args.spec, dag, root)
        print(f"Wrote DAG to {out_path}")

    if args.mermaid:
        out_path = Path.cwd() / "value_tree_dag.mmd"
        direction = "TD" if args.vertical else "LR"
        text = render_dag_mermaid(dag, root, direction=direction)
        out_path.write_text(text)
        print(f"Wrote Mermaid DAG to {out_path}")

    return
