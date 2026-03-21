import pathlib

import cyclopts
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.gridworld import GridWorld
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from valtr.gridworld_utils import GridWorldDriftFn, parse_rooms
from valtr.mintime_rollout import MinTimeRollout
from valtr.safety_filter import SafetyFilter
from valtr.solve_discrete import load_discrete_sol, save_discrete_sol, solve_discrete
from valtr.valtr import to_dag

plt.style.use("seaborn-v0_8-darkgrid")

app = cyclopts.App()

MAP = """
#############
#############
#############
### A#    ###
### ## ## ###
###    #B ###
#############
#############
#############
"""


def get_rooms():
    s = MAP
    dyn, d_raw = parse_rooms(s)
    d = {
        "A": np.where(d_raw["A"], 1, -1),
        "B": np.where(d_raw["B"], 1, -1),
        "w": np.where(d_raw["#"], 1, -1),
    }
    return dyn, d


@app.default()
def main():
    results_dir = pathlib.Path("plots_discrete")
    results_dir.mkdir(exist_ok=True)

    # TASK_SOURCE = "F A && F B && G( !w )"
    TASK_SOURCE = "G( !w )"

    logger.info("Generating the value tree DAG from logic...")
    print(f"Input task logic: {TASK_SOURCE}")

    value_tree_dag, dag_root = to_dag(TASK_SOURCE, ir_filename="rooms_discrete_ir", dag_filename="rooms_discrete_dag")
    dag_nodes = value_tree_dag.nodes

    dyn: GridWorld
    dyn, dict_predicates_unflat = get_rooms()
    dict_predicates = {k: v.flatten() for k, v in dict_predicates_unflat.items()}

    h, w = dyn.shape
    _, d_raw = parse_rooms(MAP)

    # Rename.
    d_raw["w"] = d_raw.pop("#")

    key_tmp = list(d_raw.keys())[0]
    empty_map = np.zeros_like(d_raw[key_tmp])
    for ii, (k, v) in enumerate(d_raw.items()):
        empty_map = np.where(v, ii, empty_map)

    tick_locs = np.arange(len(d_raw)) + 0.5

    # -------------------------------------------
    # Solve.
    dict_vars, dict_actions, dict_GU_vars, dict_GU_actions = solve_discrete(dyn, dag_nodes, dict_predicates)

    # ----------------------------
    # Start at the center.
    state = dyn.encode_state((5, 6))
    a_nom = dyn.str_to_action("L")

    Tp1_state = [state]

    safety_filter = SafetyFilter(dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)

    # Rollout safety filter.
    for kk in range(10):
        a_safe = safety_filter.filter_action(state, a_nom)
        state = dyn.step(state, a_safe)
        Tp1_state.append(state)

    # -----------------------------------
    # Visualize.
    cmap = plt.get_cmap("tab20", len(d_raw))
    colors = cmap.colors
    if " " in d_raw:
        space_idx = list(d_raw.keys()).index(" ")
        colors[space_idx] = np.array([1.0, 1.0, 1.0, 0.0])
    if "w" in d_raw:
        space_idx = list(d_raw.keys()).index("w")
        colors[space_idx] = np.array([140 / 255, 114 / 255, 179 / 255, 0.3])

    cmap = ListedColormap(colors)

    T_x, T_y = [], []
    for state in Tp1_state:
        x, y = dyn.decode_state(state, which=np)
        T_y.append(y)
        T_x.append(x)

    T_state_xy = np.stack([T_x, T_y], axis=1)
    print("State:")
    print(T_state_xy)

    fig, ax = plt.subplots()
    im = ax.imshow(empty_map.T, cmap=cmap, vmin=0, vmax=len(d_raw))
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))

    ax.grid(which="major", visible=False)
    ax.grid(which="minor", visible=True)

    ax.set_xticks(np.arange(h + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(w + 1) - 0.5, minor=True)

    lim = 0.2
    T_offset = np.linspace(-lim, lim, num=20)[: len(T_x)]

    T_x = T_x + T_offset
    T_y = T_y - T_offset

    ax.plot(T_x, T_y, marker="o", color="C1")
    ax.plot(T_x[0], T_y[0], marker="s", color="C1")

    ax.set_title(r"$G(\neg w)$")

    fig_path = results_dir / "run_safety_filter.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-2)
    plt.close(fig)
    logger.success(f"Saved to {fig_path}")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
