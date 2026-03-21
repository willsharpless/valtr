import pathlib

import cyclopts
import einops as ei
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.discrete import ActionInt, StateInt
from dvi.dynamics.gridworld import GridWorld
from dvi.dynamics.gridworld_timed import GridWorldTimed
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from valtr.dag_graphviz import visualize_dag
from valtr.gridworld_utils import GridWorldDriftFn, parse_rooms
from valtr.mintime_rollout import MinTimeRollout
from valtr.safety_filter import SafetyFilter
from valtr.solve_discrete import load_discrete_sol, save_discrete_sol, solve_discrete
from valtr.valtr import to_dag

plt.style.use("seaborn-v0_8-darkgrid")

app = cyclopts.App()


def _setup_ax(ax, h, w):
    ax.set_aspect("equal")
    ax.grid(which="major", visible=False)
    ax.grid(which="minor", visible=True)
    ax.set_xticks(np.arange(h + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(w + 1) - 0.5, minor=True)


def _make_overlay_text(ax, x, y, ha, va):
    return ax.text(
        x,
        y,
        "",
        transform=ax.transAxes,
        verticalalignment=va,
        horizontalalignment=ha,
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )


MAP = """
#############
#       v   #
#           #
#           #
#   sss sss #
#   sSs sWs #
#   sssssss #
#           #
#############
"""

TMAX = 30


def get_rooms():
    s = MAP
    dyn, d_raw = parse_rooms(s)
    d = {
        "site": np.where(d_raw["s"] | d_raw["S"] | d_raw["W"], 1, -1),
        "S": np.where(d_raw["S"], 1, -1),
        "W": np.where(d_raw["W"], 1, -1),
        "v": np.where(d_raw["v"], 1, -1),
        "w": np.where(d_raw["#"], 1, -1),
    }
    return dyn, d


@app.default()
def main():
    results_dir = pathlib.Path("plots_discrete")
    results_dir.mkdir(exist_ok=True)

    TAIL = "(!w) U (G( ( (site && !w && Tle20) U (site && !w && S && Tle20)) && ( (site && !w) U (site && !w && W && Tle20)) ))"
    TASK_SOURCE = "((!site && !w) U (v && !w && Tle10)) && ({})".format(TAIL)
    # TASK_SOURCE = TAIL

    logger.info("Generating the value tree DAG from logic...")
    print(f"Input task logic: {TASK_SOURCE}")

    to_dag(TAIL, dag_filename="rooms_discrete_dag_dbg_tail")

    value_tree_dag, dag_root = to_dag(
        TASK_SOURCE, ir_filename="rooms_discrete_ir", dag_filename="rooms_discrete_dbg_dag"
    )

    visualize_dag(value_tree_dag, dag_root, filename="rooms_discrete_dbg_noavoid_dag", view=False, hide_avoid=True)

    dag_nodes = value_tree_dag.nodes

    dyn_: GridWorld
    dyn_, dict_predicates_unflat = get_rooms()
    dict_predicates = {k: v.flatten() for k, v in dict_predicates_unflat.items()}

    def should_reset_timer(t: int, s: StateInt, a: ActionInt, t_next: int, s_next: StateInt):
        at_S = jnp.array(dict_predicates["S"])[s_next] > 0
        at_W = jnp.array(dict_predicates["W"])[s_next] > 0
        should_reset = (at_S | at_W) & (t_next >= TMAX - 2)
        return should_reset

    h, w = dyn_.shape
    _, d_raw = parse_rooms(MAP)

    dyn = GridWorldTimed(
        dyn_.shape, TMAX, drift_fn=dyn_.drift_fn, freeze_at_t_max=False, should_reset_timer=should_reset_timer
    )

    # Make predicates timed.
    dict_predicates_timed = {k: ei.repeat(v, "x -> (x T)", T=TMAX + 1) for k, v in dict_predicates.items()}
    dict_predicates_timed["Tle10"] = dyn.tle_predicate(10)
    dict_predicates_timed["Tle20"] = dyn.tle_predicate(20)

    # Rename.
    d_raw["w"] = d_raw.pop("#")

    key_tmp = list(d_raw.keys())[0]
    empty_map = np.zeros_like(d_raw[key_tmp])
    for ii, (k, v) in enumerate(d_raw.items()):
        empty_map = np.where(v, ii, empty_map)

    tick_locs = np.arange(len(d_raw)) + 0.5

    # -------------------------------------------
    # Solve.
    dict_vars, dict_actions, dict_GU_vars, dict_GU_actions = solve_discrete(dyn, dag_nodes, dict_predicates_timed)

    # ----------------------------
    # Visualize the value function at the root.
    T_dict_vars = {k: v.reshape(dyn.shape) for k, v in dict_vars.items()}
    T_dict_actions = {k: v.reshape(dyn.shape) for k, v in dict_actions.items()}
    T_val_root = T_dict_vars[dag_root]

    times = [0, 8, 9, 10, 11, 12]
    nrow = len(times)
    figsize = np.array([6, 2 * nrow])
    fig, axes = plt.subplots(nrow, layout="constrained", figsize=figsize)
    for ii, ax in enumerate(axes):
        val_root = T_val_root[..., times[ii]]
        im = ax.imshow(val_root.T, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_title("Root Value function, t={}".format(times[ii]))
        _setup_ax(ax, h, w)

    fig.savefig(results_dir / "safety_filter_dbg_val_root.pdf")

    # if T_val_root[:, :, 0].max() < 0:
    #     logger.warning("Value function at root is negative.")
    ipdb.set_trace()

    # ----------------------------
    # Start at the center.
    state = dyn.encode_state((2, 5, 0))
    a_nom = dyn.str_to_action(".")

    Tp1_states = [state]
    T_a_nom = []
    T_a_filt = []
    T_hasfiltered = []

    safety_filter = SafetyFilter(dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)

    # Rollout safety filter.
    for kk in range(2 * TMAX):
        logger.debug(f"> kk={kk}")
        a_safe = safety_filter.filter_action(state, a_nom)
        hasfiltered = a_safe != a_nom
        state = dyn.step(state, a_safe)
        T_a_nom.append(a_nom)
        T_a_filt.append(a_safe)
        T_hasfiltered.append(hasfiltered)
        Tp1_states.append(state)

    T_hasfiltered = np.array(T_hasfiltered)

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

    # -----------------------------------
    # Animate the rollout.
    n_frames = len(Tp1_states) - 1
    fig, ax = plt.subplots()
    im = ax.imshow(empty_map.T, cmap=cmap, vmin=0, vmax=len(d_raw))
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))
    _setup_ax(ax, h, w)

    (agent_dot,) = ax.plot([], [], marker="o", color="C0", ms=5, linestyle="None")
    kk_text = _make_overlay_text(ax, 0.02, 0.98, ha="left", va="top")
    debug_text = ax.text(
        0.98,
        0.02,
        "",
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        color="white",
        fontsize=8,
        fontname="DejaVu Sans",
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )

    def init_fn():
        return [agent_dot, kk_text, debug_text]

    def update_fn(kk: int) -> list[plt.Artist]:
        state = Tp1_states[kk]
        x, y, t = dyn.decode_state(state)
        agent_dot.set_data([x], [y])
        kk_text.set_text(f"Step {kk: 3}")
        a_nom_str = dyn.action_to_str(T_a_nom[kk])
        a_safe_str = dyn.action_to_str(T_a_filt[kk])
        has_filtered = int(T_hasfiltered[kk])
        debug_text.set_text(
            "Timer : {}\nNom : {}\nSafe: {}\nfiltered: {}".format(t, a_nom_str, a_safe_str, has_filtered)
        )
        return [agent_dot, kk_text, debug_text]

    anim = FuncAnimation(fig, update_fn, n_frames, init_fn, blit=True)
    anim.save("safety_filter_dbg_rollout.mp4", fps=5, dpi=200)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
