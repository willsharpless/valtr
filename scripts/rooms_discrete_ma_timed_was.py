import functools as ft
import pathlib

import cyclopts
import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.gridworld import GridWorld
from dvi.dynamics.gridworld_ma import (
    GridWorldMA,
    flat_totimed,
    ma_collision_predicate,
    ma_distance_predicate,
    rew_to_ma,
)
from dvi.dynamics.gridworld_ma_timed import GridWorldMATimed
from dvi.dynamics.gridworld_timed import GridWorldTimed
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import FancyArrowPatch

from valtr.gridworld_utils import GridWorldDriftFn, parse_rooms
from valtr.mintime_rollout import MinTimeRollout
from valtr.solve_discrete import load_discrete_sol, save_discrete_sol, solve_discrete
from valtr.valtr import to_dag

plt.style.use("seaborn-v0_8-darkgrid")

app = cyclopts.App()

MAP = """
################
#............# #
#.##.#.##.##.# #
#.#....#..FF.d #
#.EE.#....FF.d #
#.EE.###.....# #
#............# #
#####dddd##### #
#              #
# g #    ##  ###
#######       K#
#     #  #     #
#  #  #  ### ###
#s #  #  #     #
# s#     # AAA #
################
"""

TASK_SOURCE = "(!site U gear) && G F saw && G F wood && (!d U k) && G(!w) && G(!collide) && G(!distant)"

TMAX = 50


def draw_room_map(fig, ax, empty_map, cmap, d_raw, w, h, alpha: float = 0.5):
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])
    ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw), alpha=alpha, origin="lower")
    ax.set_xticks(np.arange(w + 1) - 0.5)
    ax.set_yticks(np.arange(h + 1) - 0.5)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.margins(0)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)


def make_room_cmap(d_raw_viz: dict[str, np.ndarray]) -> tuple[np.ndarray, ListedColormap]:
    h, w = next(iter(d_raw_viz.values())).shape[:2]
    empty_mask = np.zeros((h, w), dtype=bool)
    get_mask = lambda key: d_raw_viz.get(key, empty_mask)

    empty_map = np.zeros((h, w))
    space_idx = list(d_raw_viz.keys()).index(" ") if " " in d_raw_viz else 0
    for ii, (key, value) in enumerate(d_raw_viz.items()):
        mask = value[:, :, -1] if value.ndim == 3 else value
        if key == "s":
            empty_map = np.where(mask, space_idx, empty_map)
        else:
            empty_map = np.where(mask, ii, empty_map)

    cmap = plt.get_cmap("tab10", len(d_raw_viz))
    colors = np.array(cmap.colors, copy=True)

    if " " in d_raw_viz:
        colors[list(d_raw_viz.keys()).index(" ")] = np.array([1.0, 1.0, 1.0, 0.0])

    if "." in d_raw_viz:
        colors[list(d_raw_viz.keys()).index(".")] = np.array([227 / 255, 197 / 255, 87 / 255, 0.5])

    if "s" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("s")] = colors[space_idx]

    if "A" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("A")] = np.array([140 / 255, 114 / 255, 179 / 255, 1.0])

    if "B" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("B")] = np.array([147 / 255, 120 / 255, 96 / 255, 1.0])

    if "C" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("C")] = np.array([221 / 255, 132 / 255, 83 / 255, 1.0])

    if "E" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("E")] = np.array([85 / 255, 168 / 255, 104 / 255, 1.0])

    if "F" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("F")] = np.array([0.8, 0.4, 0.4, 1.0])

    if "K" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("K")] = np.array([221 / 255, 132 / 255, 83 / 255, 1.0])

    if "d" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("d")] = np.array([0.05, 0.05, 0.05, 1.0])

    if "g" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("g")] = np.array([77 / 255, 114 / 255, 176 / 255, 1.0])

    if "#" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("#")] = np.array([0.33714769, 0.41920711, 0.54334937, 1.0])

    if "1" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("1")] = np.array([0.8, 0.4, 0.4, 1.0])

    if "2" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("2")] = np.array([147 / 255, 120 / 255, 96 / 255, 0.7])

    return empty_map, ListedColormap(colors)


@app.default()
def main(gamma: float | None = None, resolve: bool = False):
    results_dir = pathlib.Path("plots_discrete")
    results_dir.mkdir(exist_ok=True)

    logger.debug("Parsing...")
    dyn, d_raw = parse_rooms(MAP, TMAX)
    logger.debug("Parsing... Done!")

    empty_mask = np.zeros_like(d_raw["#"], dtype=bool)
    get_mask = lambda key: d_raw.get(key, empty_mask)

    d = {
        "r1": np.where(get_mask("A"), 1, -1)[:, :, -1],
        "r2": np.where(get_mask("B"), 1, -1)[:, :, -1],
        "r3": np.where(get_mask("C"), 1, -1)[:, :, -1],
        "saw": np.where(get_mask("E"), 1, -1)[:, :, -1],
        "wood": np.where(get_mask("F"), 1, -1)[:, :, -1],
        "gear": np.where(get_mask("g"), 1, -1)[:, :, -1],
        "site": np.where(get_mask("."), 1, -1)[:, :, -1],
        "k": np.where(get_mask("K"), 1, -1)[:, :, -1],
        "d": np.where(get_mask("D"), 1, -1)[:, :, -1],
        "w": np.where(get_mask("#"), 1, -1)[:, :, -1],
        "<": np.where(get_mask("<"), 1, -1)[:, :, -1],
        ">": np.where(get_mask(">"), 1, -1)[:, :, -1],
        "^": np.where(get_mask("^"), 1, -1)[:, :, -1],
    }
    if any(key in d_raw for key in ("<", ">", "^", "v")):
        dyn.drift_fn = GridWorldDriftFn({k: v[:, :, -1] for k, v in d_raw.items()}, force=False)

    dyn_untimed = dyn
    if isinstance(dyn, GridWorldTimed):
        dyn_untimed = GridWorld((dyn.shape[0], dyn.shape[1]), dyn.drift_fn)

    logger.debug("Creating GridWorldMA")
    dyn_ma_ = GridWorldMA(dyn_untimed, n_agents=2)
    logger.debug("Creating GridWorldMATimed")
    dyn_ma = GridWorldMATimed(dyn_ma_, t_max=TMAX, freeze_at_t_max=False)

    collide_dist = 1.0
    for key, value in d.items():
        assert value.ndim == 2, f"Predicate {key} should be 2D, got shape {value.shape}"

    h, w, _ = dyn.shape
    logger.debug("dyn.shape: {}", dyn.shape)

    d_raw_viz = {k: v for k, v in d_raw.items() if not k.startswith("Tle")}
    empty_map, cmap = make_room_cmap(d_raw_viz)

    fig, ax = plt.subplots(figsize=np.array([8, 6]), frameon=False)
    draw_room_map(fig, ax, empty_map, cmap, d_raw_viz, w, h, alpha=0.5)
    fig_path = results_dir / "ma_timed_was_map.pdf"
    fig.savefig(fig_path, bbox_inches="tight", pad_inches=1e-2)
    plt.close(fig)
    logger.success(f"Saved to {fig_path}")

    value_tree_dag, dag_root = to_dag(
        TASK_SOURCE, ir_filename="rooms_discrete_ma_ir", dag_filename="rooms_discrete_ma_dag"
    )
    dag_nodes = value_tree_dag.nodes

    d_flat = {key: value.flatten() for key, value in d.items()}
    logger.debug("dict predicates...")
    dict_predicates = {
        "r1": flat_totimed(rew_to_ma(d_flat["r1"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "r2": flat_totimed(rew_to_ma(d_flat["r2"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "r3": flat_totimed(rew_to_ma(d_flat["r3"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "saw": flat_totimed(rew_to_ma(d_flat["saw"], dyn_ma.n_agents, "min"), t_max=TMAX),
        "wood": flat_totimed(rew_to_ma(d_flat["wood"], dyn_ma.n_agents, "min"), t_max=TMAX),
        "gear": flat_totimed(rew_to_ma(d_flat["gear"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "site": flat_totimed(rew_to_ma(d_flat["site"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "k": flat_totimed(rew_to_ma(d_flat["k"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "d": flat_totimed(rew_to_ma(d_flat["d"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "w": flat_totimed(rew_to_ma(d_flat["w"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "collide": ma_collision_predicate(dyn_ma_, collide_dist, t_max=TMAX),
        "distant": flat_totimed(ma_distance_predicate(dyn_ma_, 2 * collide_dist), t_max=TMAX),
    }
    for key, value in dict_predicates.items():
        assert value.ndim == 1 and value.shape[0] == dyn_ma.n_states, (
            f"Predicate {key} has wrong shape {value.shape}"
        )

    pkl_path = results_dir / "rooms_discrete_ma_timed_was_sol.pkl"
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
            pkl_path, dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras=extras
        )

    dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras = load_discrete_sol(
        pkl_path
    )

    rng = np.random.default_rng(seed=12345)
    T_value = np.asarray(dict_vars[dag_root])
    value = T_value.reshape((dyn_ma_.n_states, TMAX + 1))[:, 0]

    if "s" in d_raw and np.any(d_raw["s"][:, :, -1]):
        fixed_positions = [tuple(pos) for pos in np.argwhere(d_raw["s"][:, :, -1])]
    else:
        fixed_positions = [(4, 4), (11, 11)]

    fig_values, axes_values = plt.subplots(
        1, len(fixed_positions), figsize=(6 * len(fixed_positions), 5), layout="constrained"
    )
    if len(fixed_positions) == 1:
        axes_values = [axes_values]

    root_value_maps = []
    for fixed_pos in fixed_positions:
        value_map = np.full(dyn_untimed.shape, np.nan, dtype=float)
        for y in range(h):
            for x in range(w):
                if d_raw["#"][y, x, -1]:
                    continue
                joint_state = int(dyn_ma_.encode_from_tups([(y, x), fixed_pos]))
                value_map[y, x] = float(value[joint_state])
        root_value_maps.append(value_map)

    finite_values = np.concatenate([value_map[np.isfinite(value_map)] for value_map in root_value_maps])
    vmin = float(finite_values.min()) if finite_values.size else -1.0
    vmax = float(finite_values.max()) if finite_values.size else 1.0

    im_values = None
    for ax_value, fixed_pos, value_map in zip(axes_values, fixed_positions, root_value_maps):
        im_values = ax_value.imshow(value_map, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        ax_value.set_xticks(np.arange(w + 1) - 0.5)
        ax_value.set_yticks(np.arange(h + 1) - 0.5)
        ax_value.grid(color="k", linestyle="-", linewidth=0.5, alpha=0.3)
        ax_value.set_xlim(-0.5, w - 0.5)
        ax_value.set_ylim(-0.5, h - 0.5)
        ax_value.margins(0)
        ax_value.tick_params(labelbottom=False, labelleft=False)
        ax_value.set_aspect("equal")
        ax_value.set_autoscale_on(False)
        fixed_y, fixed_x = fixed_pos
        ax_value.plot([fixed_x], [fixed_y], marker="o", color="C1", ms=7, linestyle="None")
        ax_value.set_title(f"Root value | agent 2 fixed at {fixed_pos}")

    if im_values is not None:
        fig_values.colorbar(im_values, ax=axes_values, fraction=0.046, pad=0.04)
    fig_values.savefig(results_dir / "rooms_discrete_ma_timed_was_root_values.png", dpi=200, bbox_inches="tight")
    plt.close(fig_values)

    feasible_states = np.where(value >= 0)[0]
    logger.info("Num feasible states: {}", len(feasible_states))

    is_good = (dict_predicates["k"].reshape((dyn_ma_.n_states, TMAX + 1))[:, 0] != 1) & (value >= 0)
    feasible_states_good = np.where(is_good)[0]
    logger.info("Num feasible states (where not on key): {}", len(feasible_states_good))

    start_state = None
    if "s" in d_raw and np.any(d_raw["s"][:, :, -1]):
        start_positions = [tuple(pos) for pos in np.argwhere(d_raw["s"][:, :, -1])][: dyn_ma.n_agents]
        if len(start_positions) == dyn_ma.n_agents:
            start_state_untimed = int(dyn_ma_.encode_from_tups(start_positions))
            start_state = dyn_ma.encode_timed_state(start_state_untimed, t=0)

    if start_state is not None and T_value[start_state] >= 0:
        logger.info("Using MAP-defined start state.")
    else:
        if start_state is not None:
            logger.info("MAP-defined start state not feasible: {}", T_value[start_state])
        if len(feasible_states_good) > 0:
            start_state_untimed = int(rng.choice(feasible_states_good))
        else:
            start_state_untimed = int(rng.choice(feasible_states))
        start_state = dyn_ma.encode_timed_state(start_state_untimed, t=0)

    node_id = None
    if node_id is not None:
        def get_value_at_agent1_state(s_: int, t: int):
            joint_state_ = dyn_ma_.encode_joint_state([joint_state[0], s_], which=jnp)
            timed_state_ = dyn_ma.encode_timed_state(joint_state_, t)
            return jnp.array(dict_vars[node_id])[timed_state_]

        def get_value_at_time(t: int):
            out = jax.vmap(ft.partial(get_value_at_agent1_state, t=t))(np.arange(dyn_untimed.n_states))
            assert out.shape == (dyn_untimed.n_states,)
            return out.reshape(dyn_untimed.shape)

        joint_state_flat_untimed = 6979
        joint_state = dyn_ma_.decode_joint_state(joint_state_flat_untimed)
        T_values = jax.vmap(get_value_at_time)(np.arange(TMAX + 1))

        steps_to_viz = [0, 1, 2, TMAX - 2, TMAX - 1, TMAX]
        fig, axes = plt.subplots(len(steps_to_viz), figsize=np.array([8, 3 * len(steps_to_viz)]), layout="constrained")
        for ii, ax in enumerate(axes):
            t = steps_to_viz[ii]
            ax.imshow(empty_map.T, cmap=cmap, vmin=0, vmax=len(d_raw_viz), alpha=0.5)
            mask1 = np.ma.array(T_values[t].T <= 0.0, T_values[t].T)
            outline_mask_cells(ax, mask1, inset=0.12, color="red", linewidth=1.2, origin="upper")
            ax.set_title(f"Value at time {t}")
        fig.savefig(results_dir / "ma_timed_was_values.pdf", bbox_inches="tight", pad_inches=1e-2)
        plt.close(fig)
        return

    rollouter = MinTimeRollout(dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
    Tp1_states, T_actions, T_curnode_idxs = rollouter.rollout(start_state, max_steps=100)

    n_frames = len(Tp1_states)
    fig_anim, ax_anim = plt.subplots(frameon=False)
    draw_room_map(fig_anim, ax_anim, empty_map, cmap, d_raw_viz, w, h, alpha=0.5)

    n_agents = dyn_ma.n_agents
    agent_colors = ["C0", "C1", "C2"]
    agent_dots = []
    for ii in range(n_agents):
        (dot,) = ax_anim.plot([], [], marker="o", color=agent_colors[ii], ms=5, linestyle="None")
        agent_dots.append(dot)

    kk_text = ax_anim.text(
        0.02,
        0.98,
        "",
        transform=ax_anim.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )

    def init_fn():
        return agent_dots + [kk_text]

    def update_fn(kk: int) -> list[plt.Artist]:
        joint_state = Tp1_states[kk]
        agent_states_flat, _ = dyn_ma.decode_timed_state(joint_state, which=np)
        agent_states = dyn_ma_.decode_joint_state(agent_states_flat)
        for ii in range(n_agents):
            state_i = int(agent_states[ii])
            y, x = dyn_untimed.decode_state(state_i)
            agent_dots[ii].set_data([x], [y])
        kk_text.set_text(f"Step {kk: 3}")
        return agent_dots + [kk_text]

    anim = FuncAnimation(fig_anim, update_fn, n_frames, init_fn, blit=True)
    anim.save(str(results_dir / "rooms_discrete_rollout_multiagent_timed_was.mp4"), fps=5, dpi=200)
    plt.close(fig_anim)

    fig_still, ax_still = plt.subplots(frameon=False, figsize=(4, 4))
    draw_room_map(fig_still, ax_still, empty_map, cmap, d_raw_viz, w, h, alpha=0.5)

    steps_to_plot = 50
    n_plot_steps = min(steps_to_plot, len(Tp1_states))
    Tp1_states_plot = Tp1_states[:n_plot_steps]
    T_curnode_idxs_plot = T_curnode_idxs[: max(n_plot_steps - 1, 0)]

    decoded_joint_states = []
    for joint_state in Tp1_states_plot:
        joint_state_untimed, _ = dyn_ma.decode_timed_state(int(joint_state), which=np)
        decoded_joint_states.append(dyn_ma_.decode_joint_state(int(joint_state_untimed)))
    decoded_joint_states = np.array(decoded_joint_states)

    node_cmap = plt.get_cmap("tab20", max(len(dag_nodes), 1))
    offset_radius = 0.14
    if n_agents == 1:
        agent_plot_offsets = np.zeros((1, 2))
    elif n_agents == 2:
        agent_plot_offsets = offset_radius * np.array([[1.0, 1.0], [-1.0, -1.0]]) / np.sqrt(2.0)
    else:
        offset_angles = np.linspace(np.pi / 4.0, np.pi / 4.0 + 2.0 * np.pi, n_agents, endpoint=False)
        agent_plot_offsets = offset_radius * np.column_stack((np.cos(offset_angles), np.sin(offset_angles)))

    for ii in range(n_agents):
        agent_state_traj = decoded_joint_states[:, ii]
        coords = np.array([dyn_untimed.decode_state(int(state)) for state in agent_state_traj])
        offset_y, offset_x = agent_plot_offsets[ii]
        ys = coords[:, 0] + offset_y
        xs = coords[:, 1] + offset_x
        color = agent_colors[ii]

        ax_still.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, zorder=4)
        for step_idx in range(len(xs) - 1):
            start = np.array([xs[step_idx], ys[step_idx]], dtype=float)
            end = np.array([xs[step_idx + 1], ys[step_idx + 1]], dtype=float)
            delta = end - start
            if np.allclose(delta, 0.0):
                continue
            node_color = node_cmap(int(T_curnode_idxs_plot[step_idx]))
            direction = delta / np.linalg.norm(delta)
            shrink = 0.08 * direction
            arrow = FancyArrowPatch(
                posA=tuple(start + shrink),
                posB=tuple(end - shrink),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=0.5,
                facecolor=to_rgba(node_color, alpha=1.0),
                zorder=5,
            )
            ax_still.add_patch(arrow)
        ax_still.plot(xs[0], ys[0], marker="o", color=color, ms=6, zorder=6)
        ax_still.plot(xs[-1], ys[-1], marker="s", color=color, ms=5, zorder=6)

    fig_still.savefig(results_dir / "rooms_discrete_rollout_multiagent_timed_was_paths.png", dpi=200, pad_inches=0)
    plt.close(fig_still)


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

            segments.append([(x0, y0), (x1, y0)])
            segments.append([(x1, y0), (x1, y1)])
            segments.append([(x1, y1), (x0, y1)])
            segments.append([(x0, y1), (x0, y0)])

    lc = LineCollection(segments, colors=color, linewidths=linewidth)
    ax.add_collection(lc)
    return lc


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
