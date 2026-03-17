import functools as ft
import pathlib

import cyclopts
import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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

# TASK_SOURCE = "(!D U K) && (!s U (v && g)) && F G s && G F S && G F W && G(!w) && G(!collide) && F (C || T)"

TAIL = "(!w) U G( ( (site && !w) U (site && !w && S)) && ( (site && !w) U (site && !w && W)) )"
# TAIL2 = "F G( !w && ( site U (site && S)) && ( site U (site && W)) )"

# The following are equivalent.
# TASK_SOURCE = "F G site && G F S && G F W && F (C || T) & G(!w)"
# TASK_SOURCE = "(!w) U( (C || T) && {} )".format(TAIL)

# The following are equivalent.
# TASK_SOURCE = "(!s U v) && (!s U g) && F G site && G F S && G F W && F (C || T) & G(!w)"
# TASK_SOURCE = "(!w && !site) U( ((C || T) && ( (!w && !site) U ( v && g && ({}) ) ) ) || (v && g && !w U( (C || T) && ({}) ) ) )".format(
#     TAIL, TAIL
# )
# TASK_SOURCE = "(!site U v) && (!site U g) && ({}) && F (C || T) && G(!w)".format(TAIL2)
TASK_SOURCE = "((!site && !w) U (v && !w)) && ((!site && !w) U (g && !w)) && ({}) && (!w) U ( (C || T) && !w )".format(TAIL)

# TASK_SOURCE = "G( (site U (site && S) ) && (site U (site && W) ) )"


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
    # Everything in d should be 2d.
    for k, v in d.items():
        assert v.ndim == 2

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

    for k, v in dict_predicates.items():
        assert v.ndim == 1 and v.shape[0] == dyn_ma.n_states, f"Predicate {k} has wrong shape {v.shape}"

    # -----------------------------
    # Visualize.
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

    fig_path = results_dir / "ma_timed.pdf"
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
    pkl_path = results_dir / "rooms_discrete_ma_timed_sol.pkl"

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

    # ---------------------------------
    rng = np.random.default_rng(seed=12345)
    T_value = dict_vars[dag_root]
    value = T_value.reshape((dyn_ma_.n_states, TMAX + 1))[:, 0]  # Should start at the initial time.
    feasible_states = np.where(value >= 0)[0]
    logger.info("Num feasible states: {}".format(len(feasible_states)))

    start_state_untimed = rng.choice(feasible_states)
    start_state = dyn_ma.encode_timed_state(start_state_untimed, t=0)

    # node_id = 14
    # node_id = 7
    node_id = None

    if node_id is not None:
        # -----------------------------
        def get_value_at_agent1_state(s_: int, t: int):
            joint_state_ = dyn_ma_.encode_joint_state([joint_state[0], s_], which=jnp)
            timed_state_ = dyn_ma.encode_timed_state(joint_state_, t)
            return jnp.array(dict_vars[node_id])[timed_state_]

        def get_value_at_time(t: int):
            out = jax.vmap(ft.partial(get_value_at_agent1_state, t=t))(np.arange(dyn_untimed.n_states))
            assert out.shape == (dyn_untimed.n_states,)
            out = out.reshape(dyn_untimed.shape)
            return out

        joint_state_flat_untimed = 6979
        joint_state = dyn_ma_.decode_joint_state(joint_state_flat_untimed)

        T_values = jax.vmap(get_value_at_time)(np.arange(TMAX + 1))
        assert T_values.shape == (TMAX + 1, *dyn_untimed.shape)

        # Visualize the value function for node_id at different time steps.
        steps_to_viz = [0, 1, 2, TMAX - 2, TMAX - 1, TMAX]
        # steps_to_viz = [0, 1, 50, 51, 52, 53, 54, 55, 56, TMAX-1, TMAX]
        nrow = len(steps_to_viz)
        figsize = np.array([8, 3 * nrow])
        fig, axes = plt.subplots(nrow, figsize=figsize, layout="constrained")
        for ii, ax in enumerate(axes):
            t = steps_to_viz[ii]
            ax.imshow(empty_map.T, cmap=cmap, vmin=0, vmax=len(d_raw_viz), alpha=0.5)

            # im = ax.imshow(T_values[t].T, cmap="coolwarm", vmin=-1, vmax=1, alpha=0.2)
            mask1 = np.ma.array(T_values[t].T <= 0.0, T_values[t].T)
            outline_mask_cells(ax, mask1, inset=0.12, color="red", linewidth=1.2, origin="upper")

            ax.set_title(f"Value at time {t}")
            fig.colorbar(im, ax=ax)
            setup_ax(ax)
        fig_path = results_dir / "ma_timed_values.pdf"
        fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-2)
        plt.close(fig)
        exit(0)

    # -----------------------------
    rollouter = MinTimeRollout(dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
    Tp1_states, T_actions, T_curnode_idxs = rollouter.rollout(start_state, max_steps=100)

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
    anim.save("rooms_discrete_rollout_multiagent.mp4", fps=5, dpi=200)


def outline_mask_cells(
    ax,
    mask,
    inset=0.08,
    color="k",
    linewidth=1.5,
    origin="upper",
):
    """
    Draw inset outlines around all True cells in a 2D boolean mask.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    mask : 2D array-like of bool
        True where a cell should be outlined.
    inset : float, default 0.08
        Inset from each cell edge, in cell units. Must be between 0 and 0.5.
    color : matplotlib color, default 'k'
        Outline color.
    linewidth : float, default 1.5
        Width of outline lines.
    origin : {'upper', 'lower'}, default 'upper'
        Match the origin used by imshow.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if not (0 <= inset < 0.5):
        raise ValueError("inset must satisfy 0 <= inset < 0.5")

    nrows, ncols = mask.shape
    segments = []

    for r in range(nrows):
        for c in range(ncols):
            if not mask[r, c]:
                continue

            # imshow cell bounds:
            # x in [c-0.5, c+0.5], y in [r-0.5, r+0.5]
            x0 = c - 0.5 + inset
            x1 = c + 0.5 - inset

            if origin == "upper":
                y0 = r - 0.5 + inset
                y1 = r + 0.5 - inset
            elif origin == "lower":
                y0 = r - 0.5 + inset
                y1 = r + 0.5 - inset
            else:
                raise ValueError("origin must be 'upper' or 'lower'")

            # top
            segments.append([(x0, y0), (x1, y0)])
            # right
            segments.append([(x1, y0), (x1, y1)])
            # bottom
            segments.append([(x1, y1), (x0, y1)])
            # left
            segments.append([(x0, y1), (x0, y0)])

    lc = LineCollection(segments, colors=color, linewidths=linewidth)
    ax.add_collection(lc)
    return lc


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
