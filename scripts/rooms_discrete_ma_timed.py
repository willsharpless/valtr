import pathlib

import cyclopts
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.gridworld_ma import GridWorldMA, ma_collision_predicate, rew_to_ma
from dvi.dynamics.gridworld_ma_timed import GridWorldMATimed
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from valtr.gridworld_utils import GridWorldDriftFn, parse_rooms
from valtr.mintime_rollout import MinTimeRollout
from valtr.solve_discrete import load_discrete_sol, save_discrete_sol, solve_discrete
from valtr.valtr import to_dag

plt.style.use("seaborn-v0_8-darkgrid")

app = cyclopts.App()

MAP = """
         C  .
 sss    sss .
 sSs  v sWs .
 ssssssssss .
 ###        .
 #gD        .
 ### T      .
 K          .
"""

TASK_SOURCE = "(!D U K) && (!s U (v && g)) && F G s && G F S && G F W && G(!w) && G(!collide) && F (C || T)"

TMAX = 100

@app.default()
def main(gamma: float | None = None, resolve: bool = False):
    results_dir = pathlib.Path("plots_discrete")
    results_dir.mkdir(exist_ok=True)

    logger.debug("Parsing...")
    dyn, d_raw = parse_rooms(MAP, TMAX, ignore=".")
    logger.debug("Parsing... Done!")

    logger.debug("Creating GridWorldMA")
    dyn_ma_ = GridWorldMA(dyn, n_agents=2)
    logger.debug("Creating GridWorldMATimed")
    dyn_ma = GridWorldMATimed(dyn_ma_, t_max=TMAX)

    collide_dist = 1.0  # Diagonal is safe, but not adjacent.
    d = {
        "C": np.where(d_raw["C"], 1, -1),
        "T": np.where(d_raw["T"], 1, -1),
        #
        "S": np.where(d_raw["S"], 1, -1),
        "W": np.where(d_raw["W"], 1, -1),
        #
        "s": np.where(d_raw["s"] | d_raw["S"] | d_raw["W"], 1, -1),
        #
        "K": np.where(d_raw["K"], 1, -1),
        "D": np.where(d_raw["D"], 1, -1),
        "w": np.where(d_raw["#"], 1, -1),
        "collide": ma_collision_predicate(dyn_ma_, collide_dist, t_max=TMAX),
    }

    h, w, _ = dyn.shape
    logger.debug("dyn.shape: {}".format(dyn.shape))

    d_raw_viz = d_raw.copy()
    # Remove all keys starting with "Tle"
    d_raw_viz = {k: v for k,v in d_raw_viz.items() if not k.startswith("Tle")}

    empty_map = np.zeros((h, w))
    for ii, (k, v) in enumerate(d_raw_viz.items()):
        empty_map = np.where(v[:, :, -1], ii, empty_map)
    tick_locs = np.arange(len(d_raw_viz)) + 0.5
    cmap = plt.get_cmap("tab20", len(d_raw_viz))
    colors = cmap.colors

    # Get the index of " " in the keys to set it to white.
    if " " in d_raw_viz:
        space_idx = list(d_raw_viz.keys()).index(" ")
        colors[space_idx] = np.array([0.8, 0.8, 0.8, 1.0])

    cmap = ListedColormap(colors)

    # ------------------------------

    dict_predicates = {
        "C": rew_to_ma(d["C"], dyn_ma.n_agents, "max"),

    # -----------------------------
    # Visualize.
    def setup_ax(ax_: plt.Axes):
        ax_.set_aspect("equal")
        ax_.grid(which='major', visible=False)
        ax_.grid(which='minor', visible=True)

        ax_.set_xticks(np.arange(h + 1) - 0.5, minor=True)
        ax_.set_yticks(np.arange(w + 1) - 0.5, minor=True)

    figsize = np.array([8, 6])
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.imshow(empty_map.T, cmap=cmap, vmin=0, vmax=len(d_raw_viz))
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw_viz.keys()))
    setup_ax(ax)

    fig_path = results_dir / "ma_timed.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-2)
    plt.close(fig)
    logger.success(f"Saved to {fig_path}")

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
