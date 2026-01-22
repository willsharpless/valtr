import pathlib

import cyclopts
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.gridworld import GridWorld
from emoji import emojize
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import CenteredNorm, ListedColormap, to_rgba
from PIL import Image

from valtr.gridworld_utils import GridWorldDriftFn, parse_rooms
from valtr.mintime_rollout import MinTimeRollout
from valtr.print_dag import dag_node_to_str
from valtr.reachability import DAGNode
from valtr.solve_discrete import load_discrete_sol, save_discrete_sol, solve_discrete
from valtr.util.cmap import get_BuRd_smooth, get_BuRd_trunc
from valtr.util.emoji import TwemojiSVGSource, plot_emoji
from valtr.util.path_util import get_paper_plot_dir
from valtr.valtr import to_dag

app = cyclopts.App()


@app.default()
def main(pkl_path: pathlib.Path):
    paper_plot_dir = get_paper_plot_dir()
    dyn: GridWorld
    dag_nodes: list[DAGNode]
    d_raw: dict[str, np.ndarray]
    dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras = load_discrete_sol(
        pkl_path
    )

    h, w = dyn.shape
    task_source = extras["task_source"]
    d_raw = extras["d_raw"]
    map_num = extras["map_num"]

    print(d_raw.keys())

    color_dict = {
        # "A": to_rgba("C0"),
        # "B": to_rgba("C1"),
        # "K": to_rgba("C2"),
        "#": to_rgba("C3"),
        "K": to_rgba("C1", alpha=0.8),
        "D": to_rgba("C1", alpha=0.8),
    }
    label_dict = {
        "A": "A",
        "B": "B",
        "K": ":key:",
        "D": ":door:",
    }

    key_tmp = list(d_raw.keys())[0]
    # empty_map = np.zeros_like(d_raw[key_tmp])
    empty_map = np.full((h, w, 4), fill_value=0)
    for ii, (k, v) in enumerate(d_raw.items()):
        if k in color_dict:
            empty_map = np.where(v[..., None], color_dict[k], empty_map)
        # empty_map = np.where(v, ii, empty_map)

    cmap = plt.get_cmap("tab20", len(d_raw))
    colors = cmap.colors

    # --------------------------------------------

    tick_locs = np.arange(len(d_raw)) + 0.5

    figsize = np.array([6, 4])

    fig, ax = plt.subplots(figsize=figsize, dpi=400)
    im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw))

    annotate_cell(d_raw, label_dict, ax)

    # cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    # cbar.ax.set_yticklabels(list(d_raw.keys()))
    ax.set_xticks(np.arange(w + 1) - 0.5)
    ax.set_yticks(np.arange(h + 1) - 0.5)
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)

    fig_path = paper_plot_dir / f"{map_num}_map.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-2)
    plt.close(fig)

    # -------------------------------------------------
    # Visualize the value functions for each temporal node.
    cmap = get_BuRd_smooth()

    for dag_id, dag_node in enumerate(dag_nodes):
        if not dag_node.is_temporal():
            continue

        value_im = dict_vars[dag_id].reshape(dyn.shape)

        fig, ax = plt.subplots(figsize=figsize, dpi=400)
        norm = CenteredNorm()
        im = ax.imshow(value_im, cmap=cmap, alpha=0.9, origin="lower", vmin=-1, vmax=1)
        cbar = fig.colorbar(im, ax=ax)
        annotate_cell(d_raw, label_dict, ax)

        ax.set_xticks(np.arange(w + 1) - 0.5)
        ax.set_yticks(np.arange(h + 1) - 0.5)
        ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
        # ax.set_title(dag_node_to_str(dag_nodes, dag_id))
        ax.set_title(f"DAG Node {dag_id}")

        node_type = type(dag_node).__name__[3:]
        fig_path = paper_plot_dir / f"{map_num}_node_{dag_id}_{node_type}.pdf"
        fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-2)
        plt.close(fig)


def annotate_cell(d_raw: dict[str, np.ndarray], label_dict: dict[str, str], ax: plt.Axes):
    for k, v in d_raw.items():
        if k not in label_dict:
            continue

        label = label_dict[k]
        is_emoji = label.startswith(":") and label.endswith(":")
        ys, xs = np.where(v)
        for x, y in zip(xs, ys):
            if is_emoji:
                plot_emoji(
                    np.array([x, y]),
                    size_data=0.8,
                    emoji_str=label_dict[k],
                    size=512,
                    ax=ax,
                    extent="lower",
                )
            else:
                ax.text(
                    x,
                    y,
                    label_dict[k],
                    color="black",
                    fontsize=20,
                    ha="center",
                    va="center",
                )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
