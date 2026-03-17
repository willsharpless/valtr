import functools as ft
import pathlib

import cyclopts
import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.discrete import ActionInt, StateInt
from dvi.dynamics.gridworld import GridWorld
from dvi.dynamics.gridworld_ma import (GridWorldMA, flat_totimed, ma_collision_predicate, ma_distance_predicate,
                                       rew_to_ma)
from dvi.dynamics.gridworld_ma_timed import GridWorldMATimed
from dvi.dynamics.gridworld_timed import GridWorldTimed
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from valtr.gridworld_utils import GridWorldDriftFn, parse_rooms
from valtr.mintime_rollout import MinTimeRollout
from valtr.safety_filter import SafetyFilter
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

# TASK_SOURCE = "G( leash && !collide )"
# TASK_SOURCE = "G( !collide )"

TAIL = "(!w) U G( ( (site && !w) U (site && !w && S)) && ( (site && !w) U (site && !w && W)) )"
TASK_SOURCE = "((!site && !w) U (v && !w)) && ((!site && !w) U (g && !w)) && ({}) && (!w) U ( (C || T) && !w )".format(TAIL)

TMAX = 50


@app.default()
def main(gamma: float | None = None, resolve: bool = False):
    results_dir = pathlib.Path("plots_discrete")
    results_dir.mkdir(exist_ok=True)

    logger.debug("Parsing...")
    dyn, d_raw = parse_rooms(MAP, TMAX, ignore=".")
    logger.debug("Parsing... Done!")

    dyn_untimed = dyn
    if isinstance(dyn, GridWorldTimed):
        dyn_untimed = GridWorld((dyn.shape[0], dyn.shape[1]), dyn.drift_fn)

    logger.debug("Creating GridWorldMA")
    dyn_ma_ = GridWorldMA(dyn_untimed, n_agents=2)
    logger.debug("Creating GridWorldMATimed")
    dyn_ma = GridWorldMATimed(dyn_ma_, t_max=TMAX, freeze_at_t_max=False)

    collide_dist = 1.0  # Diagonal is safe, but not adjacent.
    d = {
        "C": np.where(d_raw["C"], 1, -1)[:, :, -1],
        "T": np.where(d_raw["T"], 1, -1)[:, :, -1],
        #
        "S": np.where(d_raw["S"], 1, -1)[:, :, -1],
        "W": np.where(d_raw["W"], 1, -1)[:, :, -1],
        #
        "site": np.where(d_raw["s"] | d_raw["S"] | d_raw["W"], 1, -1)[:, :, -1],
        #
        "v": np.where(d_raw["v"], 1, -1)[:, :, -1],
        "g": np.where(d_raw["g"], 1, -1)[:, :, -1],
        #
        "K": np.where(d_raw["K"], 1, -1)[:, :, -1],
        "D": np.where(d_raw["D"], 1, -1)[:, :, -1],
        "w": np.where(d_raw["#"], 1, -1)[:, :, -1],
    }

    h, w, _ = dyn.shape
    logger.debug("dyn.shape: {}".format(dyn.shape))

    d_raw_viz = d_raw.copy()
    # Remove all keys starting with "Tle"
    d_raw_viz = {k: v for k, v in d_raw_viz.items() if not k.startswith("Tle")}

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

    dict_predicates_unflat = d
    d_flat = {k: v.flatten() for k, v in dict_predicates_unflat.items()}

    logger.debug("dict predicates...")
    dict_predicates = {
        "C": flat_totimed(rew_to_ma(d_flat["C"], dyn_ma.n_agents, mode=0), t_max=TMAX),
        "T": flat_totimed(rew_to_ma(d_flat["T"], dyn_ma.n_agents, mode=0), t_max=TMAX),
        #
        "S": flat_totimed(rew_to_ma(d_flat["S"], dyn_ma.n_agents, mode=1), t_max=TMAX),
        "W": flat_totimed(rew_to_ma(d_flat["W"], dyn_ma.n_agents, mode=1), t_max=TMAX),
        #
        "site": flat_totimed(rew_to_ma(d_flat["site"], dyn_ma.n_agents, mode=1), t_max=TMAX),
        #
        "v": flat_totimed(rew_to_ma(d_flat["v"], dyn_ma.n_agents, mode=1), t_max=TMAX),
        "g": flat_totimed(rew_to_ma(d_flat["g"], dyn_ma.n_agents, mode=1), t_max=TMAX),
        #
        "w": flat_totimed(rew_to_ma(d_flat["w"], dyn_ma.n_agents, "min"), t_max=TMAX),
        #
        "collide": ma_collision_predicate(dyn_ma_, collide_dist, t_max=TMAX),
        "leash": ma_collision_predicate(dyn_ma_, 4, t_max=TMAX),
    }
    dict_predicates["w"] = dict_predicates["w"] | ~dict_predicates["leash"]

    # leash = dict_predicates["leash"]
    # logger.debug("leash min: {}, max: {}".format(leash.min(), leash.max()))
    # tmp = leash.reshape((dyn_ma_.shape * dyn_ma_.n_agents) + (TMAX + 1,))[..., 0]
    # print(tmp[0, 0, :, :])
    # exit(0)

    for k, v in dict_predicates.items():
        assert v.ndim == 1 and v.shape[0] == dyn_ma.n_states, f"Predicate {k} has wrong shape {v.shape}"

    # -----------------------------
    # Visualize map.
    def setup_ax(ax_: plt.Axes):
        ax_.set_aspect("equal")
        ax_.grid(which="major", visible=False)
        ax_.grid(which="minor", visible=True)

        ax_.set_xticks(np.arange(h + 1) - 0.5, minor=True)
        ax_.set_yticks(np.arange(w + 1) - 0.5, minor=True)

    figsize = np.array([8, 6])
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.imshow(empty_map.T, cmap=cmap, vmin=0, vmax=len(d_raw_viz))
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw_viz.keys()))
    setup_ax(ax)

    fig_path = results_dir / "safety_filter_ma_map.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-2)
    plt.close(fig)
    logger.success(f"Saved to {fig_path}")

    # -----------------------------
    # Decompose.
    value_tree_dag, dag_root = to_dag(
        TASK_SOURCE, ir_filename="rooms_discrete_ma_ir", dag_filename="rooms_discrete_ma_dag"
    )
    dag_nodes = value_tree_dag.nodes

    # -----------------------------
    # Solve.
    pkl_path = results_dir / "run_safety_filter_ma_timed_sol.pkl"

    if resolve or not pkl_path.exists():
        dict_vars, dict_actions, dict_GU_vars, dict_GU_actions = solve_discrete(
            dyn_ma, dag_nodes, dict_predicates, gamma=gamma
        )
        extras = {
            "task_source": TASK_SOURCE,
            "dict_predicates": dict_predicates,
            "gamma": gamma,
            "d_raw": d_raw,
        }
        save_discrete_sol(
            pkl_path,
            dyn_ma,
            dag_nodes,
            dag_root,
            dict_vars,
            dict_actions,
            dict_GU_vars,
            dict_GU_actions,
            extras=extras,
        )

    dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras = load_discrete_sol(
        pkl_path
    )

    # -----------------------------
    rng = np.random.default_rng(seed=12345)
    T_value = dict_vars[dag_root]

    start_state_untimed = dyn_ma_.encode_from_tups([(5, 4), (7, 4)])
    start_state = dyn_ma.encode_timed_state(start_state_untimed, t=0)

    logger.debug("Initial value: {}".format(T_value[start_state]))
    logger.debug("Initial leash: {}".format(dict_predicates["leash"][start_state]))

    dyn_ma: GridWorldMATimed
    # a_nom = dyn_ma_.str_to_action("L|.")
    a_nom = dyn_ma_.str_to_action(".|U")

    safety_filter = SafetyFilter(dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)

    state = start_state
    Tp1_states = [state]
    T_hasfiltered = []

    def preference_fn(_: StateInt, a_nom_: ActionInt) -> np.ndarray:
        # If agent1 is staying still, then prefer actions where the agent2 action matches a_nom_.
        # Otherwise, prefer actions where agent1 action matches a_nom_.
        N_actions = dyn_ma_.decode_joint_action(a_nom_, which=np)
        assert N_actions.shape == (2,)
        a1_nom, a2_nom = N_actions

        STILL_ACTION = dyn_ma_.base.str_to_action(".")

        costs = np.zeros(dyn_ma.n_actions, dtype=np.float32)
        for a in range(dyn_ma.n_actions):
            N_a = dyn_ma_.decode_joint_action(a, which=np)
            a1, a2 = N_a

            if a1_nom == STILL_ACTION:  # Agent 1 is staying still
                # Prefer actions where agent 2 matches a_nom_
                costs[a] = 0.0 if a2 == a2_nom else 1.0
            else:
                # Prefer actions where agent 1 matches a_nom_
                costs[a] = 0.0 if a1 == a1_nom else 1.0

        return costs

    # Rollout safety filter.
    for kk in range(10):
        a_safe = safety_filter.filter_action(state, a_nom, preference_fn)
        hasfiltered = a_safe != a_nom
        state = dyn_ma.step(state, a_safe)
        Tp1_states.append(state)
        T_hasfiltered.append(hasfiltered)

    T_hasfiltered = np.array(T_hasfiltered)

    # ---------------------------------------------------------------------
    # Visualize the rollout by animating the path and saving as mp4.
    n_frames = len(Tp1_states)
    fig, ax = plt.subplots()

    # Visualize the map again.
    im = ax.imshow(empty_map.T, cmap=cmap, vmin=0, vmax=len(d_raw_viz), alpha=0.5)
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw_viz.keys()))
    setup_ax(ax)

    # --- Multi-agent: one dot per agent ---
    n_agents = dyn_ma.n_agents
    agent_colors = ["C0", "C1", "C2"]

    agent_dots = []
    for i in range(n_agents):
        (dot,) = ax.plot([], [], marker="o", color=agent_colors[i], ms=5, linestyle="None")
        agent_dots.append(dot)

    kk_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )

    def init_fn():
        # Return all animated artists
        return agent_dots + [kk_text]

    def update_fn(kk: int) -> list[plt.Artist]:
        joint_state = Tp1_states[kk]

        # Decode joint state -> per-agent single-agent states (N,)
        agent_states_flat, tt = dyn_ma.decode_timed_state(joint_state, which=np)
        agent_states = dyn_ma_.decode_joint_state(agent_states_flat)  # (N,)

        # Update each agent dot
        for i in range(n_agents):
            s_i = int(agent_states[i])  # safe for matplotlib
            x, y = dyn_untimed.decode_state(s_i)  # (y, x) for your gridworld
            agent_dots[i].set_data([x], [y])

        kk_text.set_text(f"Step {kk: 3}")

        return agent_dots + [kk_text]

    anim = FuncAnimation(fig, update_fn, n_frames, init_fn, blit=True)
    anim.save("safety_filter_ma_rollout.mp4", fps=5, dpi=200)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
