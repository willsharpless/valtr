import functools as ft
import pathlib
from collections import Counter, deque

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
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.patches import FancyArrowPatch

from valtr.filtered_rollout import FilteredRollout
from valtr.gridworld_utils import GridWorldDriftFn, parse_rooms
from valtr.mintime_policy import MinTimePolicy
from valtr.mintime_rollout import MinTimeRollout
from valtr.safety_filter import SafetyFilter
from valtr.solve_discrete import load_discrete_sol, save_discrete_sol, solve_discrete
from valtr.valtr import to_dag

plt.style.use("seaborn-v0_8-darkgrid")

app = cyclopts.App()

# MAP = """
# ################
# #......#  #  s #
# #.##EE.# g#   s#
# #.#.EE.#  # ####
# #......# ##    #
# #.#.##.d  #    #
# #....#.d  #### #
# #.##.#.d       #
# #.#....d       #
# #......# # #####
# #.#FF..# #  #  #
# #.#FF..#    # A#
# #......#      A#
# ###dd### #  # A#
# #        #K #  #
# ################
# """

MAP = """
############
#EE.#gg# s #
#EE.#gg#  s#
###.#  ### #
#A..# ##   #
#.#.d      #
#...# # ####
#FF.# # #  #
#FF.#   ^ A#
##d## # # A#
#B    #K#  #
############
"""

## old version
# TASK_SOURCE = TASK_SOURCE_AG1 = TASK_SOURCE_AG2 = "(!site U gear) && G F saw && G F wood && (!d U k) && G(!w) && G(!collide) && G(!distant)" # old

## base spec
TAIL = "(!w) U G( ( (site && !w) U (site && !w && saw)) && ( (site && !w) U (site && !w && wood)) )"
# TASK_SOURCE = "((!site && !w) U (gear && !w)) && (!d U k) && ({}) && (!w) U ( (r1 || r2) && !w )".format(TAIL)
TASK_SOURCE = "((!site && !w) U (gear && !w)) && (!d U k) && ({})".format(TAIL)

## agent 1 -> coffee
TASK_SOURCE_AG1 = "(!w U r1) && ((!w && !d) U k) && ((!w && !site) U gear)"
TASK_SOURCE_AG2 = "((!site && !w) U (gear && !w)) && (!d U k) && ({})".format(TAIL)

## agent 1 -> tea
# TASK_SOURCE_AG1 = "(!w U r2) && ((!w && !d) U k) && ((!w && !site) U gear)"
# TASK_SOURCE_AG2 = "((!site && !w) U (gear && !w)) && (!d U k) && ({})".format(TAIL)

TMAX = 100

RESULTS_DIR = pathlib.Path("plots_discrete")
def _solve_and_cache(dyn_ma, dag_nodes, dag_root, dict_predicates, pkl_path, task_source, d_raw, gamma, resolve):
    RESULTS_DIR.mkdir(exist_ok=True)
    if resolve or not pkl_path.exists():
        dict_vars, dict_actions, dict_GU_vars, dict_GU_actions = solve_discrete(
            dyn_ma, dag_nodes, dict_predicates, gamma=gamma
        )
        extras = {"task_source": task_source, "dict_predicates": dict_predicates, "gamma": gamma, "d_raw": d_raw}
        save_discrete_sol(
            pkl_path, dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras=extras
        )
    return load_discrete_sol(pkl_path)

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

    arrow_specs = {
        ">": ((-0.33, 0.0), (0.33, 0.0)),
        "<": ((0.33, 0.0), (-0.33, 0.0)),
        "^": ((0.0, -0.33), (0.0, 0.33)),
        "v": ((0.0, 0.33), (0.0, -0.33)),
    }

    for key, (start_delta, end_delta) in arrow_specs.items():
        if key not in d_raw:
            continue

        mask = d_raw[key][:, :, -1] if d_raw[key].ndim == 3 else d_raw[key]
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            arrow = FancyArrowPatch(
                posA=(x + start_delta[0], y + start_delta[1]),
                posB=(x + end_delta[0], y + end_delta[1]),
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.0,
                edgecolor="black",
                facecolor="none",
                fill=False,
                zorder=3,
            )
            ax.add_patch(arrow)


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

    if "^" in d_raw_viz:
        # colors[list(d_raw_viz.keys()).index("^")] = np.array([227 / 255, 197 / 255, 87 / 255, 0.5])
        colors[list(d_raw_viz.keys()).index("^")] = np.array([209 / 255, 119 / 255, 143 / 255, 1.])

    if "." in d_raw_viz:
        colors[list(d_raw_viz.keys()).index(".")] = np.array([227 / 255, 197 / 255, 87 / 255, 0.5])

    if "s" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("s")] = colors[space_idx]

    if "A" in d_raw_viz:
        # colors[list(d_raw_viz.keys()).index("A")] = np.array([140 / 255, 114 / 255, 179 / 255, 1.0])
        # colors[list(d_raw_viz.keys()).index("A")] = np.array([147 / 255, 120 / 255, 96 / 255, 1.0])
        colors[list(d_raw_viz.keys()).index("A")] = np.array([128 / 255, 101 / 255, 79 / 255, 1.0])

    if "B" in d_raw_viz:
        # colors[list(d_raw_viz.keys()).index("B")] = np.array([147 / 255, 120 / 255, 96 / 255, 1.0])
        # colors[list(d_raw_viz.keys()).index("B")] = np.array([140 / 255, 114 / 255, 179 / 255, 1.0])
        colors[list(d_raw_viz.keys()).index("B")] = np.array([135 / 255, 87 / 255, 207 / 255, 1.0])

    if "C" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("C")] = np.array([221 / 255, 132 / 255, 83 / 255, 1.0])

    if "E" in d_raw_viz:
        # colors[list(d_raw_viz.keys()).index("E")] = np.array([85 / 255, 168 / 255, 104 / 255, 1.0])
        colors[list(d_raw_viz.keys()).index("E")] = np.array([76 / 255, 169 / 255, 97 / 255, 1.0])

    if "F" in d_raw_viz:
        # colors[list(d_raw_viz.keys()).index("F")] = np.array([0.8, 0.4, 0.4, 1.0])
        colors[list(d_raw_viz.keys()).index("F")] = np.array([0.78, 0.17, 0.17, 1.0])

    if "K" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("K")] = np.array([221 / 255, 132 / 255, 83 / 255, 1.0])

    if "d" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("d")] = np.array([0.05, 0.05, 0.05, 1.0])

    if "g" in d_raw_viz:
        # colors[list(d_raw_viz.keys()).index("g")] = np.array([0.302, 0.447, 0.690, 1.0])
        # colors[list(d_raw_viz.keys()).index("g")] = np.array([0.361, 0.518, 0.804, 1.0])
        colors[list(d_raw_viz.keys()).index("g")] = np.array([49 / 255, 107 / 255, 207 / 255, 1.0])

    if "#" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("#")] = np.array([0.33714769, 0.41920711, 0.54334937, 1.0])

    if "1" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("1")] = np.array([0.8, 0.4, 0.4, 1.0])

    if "2" in d_raw_viz:
        colors[list(d_raw_viz.keys()).index("2")] = np.array([147 / 255, 120 / 255, 96 / 255, 0.7])

    return empty_map, ListedColormap(colors)


def decode_timed_joint_states(
    Tp1_states: list[int] | np.ndarray,
    dyn_ma: GridWorldMATimed,
    dyn_ma_untimed: GridWorldMA,
) -> np.ndarray:
    decoded_joint_states = []
    for joint_state in Tp1_states:
        joint_state_untimed, _ = dyn_ma.decode_timed_state(int(joint_state), which=np)
        decoded_joint_states.append(dyn_ma_untimed.decode_joint_state(int(joint_state_untimed), which=np))
    return np.asarray(decoded_joint_states)


def get_agent_plot_offsets(n_agents: int, offset_radius: float = 0.14) -> np.ndarray:
    if n_agents == 1:
        return np.zeros((1, 2))
    if n_agents == 2:
        return offset_radius * np.array([[1.0, 1.0], [-1.0, -1.0]]) / np.sqrt(2.0)

    offset_angles = np.linspace(np.pi / 4.0, np.pi / 4.0 + 2.0 * np.pi, n_agents, endpoint=False)
    return offset_radius * np.column_stack((np.cos(offset_angles), np.sin(offset_angles)))


def compute_plot_segments(T_curnode_idxs: list[int] | np.ndarray, max_segment_plots: int = 4) -> list[tuple[int, int]]:
    n_actions = len(T_curnode_idxs)
    if n_actions == 0:
        return []

    switch_starts = [idx for idx in range(1, n_actions) if T_curnode_idxs[idx] != T_curnode_idxs[idx - 1]]
    segment_starts = [0] + switch_starts

    segments = []
    for seg_idx, start in enumerate(segment_starts):
        end = segment_starts[seg_idx + 1] if seg_idx + 1 < len(segment_starts) else n_actions
        segments.append((start, end))

    if len(segments) <= max_segment_plots:
        return segments

    return segments[: max_segment_plots - 1] + [(segments[max_segment_plots - 1][0], n_actions)]


def save_rollout_still(
    fig_path: pathlib.Path,
    decoded_joint_states: np.ndarray,
    state_start: int,
    state_end: int,
    node_ids: list[int] | np.ndarray,
    draw_arrows: bool,
    fig_size: tuple[float, float],
    empty_map: np.ndarray,
    cmap: ListedColormap,
    d_raw_viz: dict[str, np.ndarray],
    w: int,
    h: int,
    dyn_untimed: GridWorld,
    agent_colors: list[str],
):
    fig_still, ax_still = plt.subplots(frameon=False, figsize=fig_size)
    draw_room_map(fig_still, ax_still, empty_map, cmap, d_raw_viz, w, h, alpha=0.5)

    n_agents = decoded_joint_states.shape[1]
    node_cmap = plt.get_cmap("tab20", max(len(node_ids), 1))
    agent_plot_offsets = get_agent_plot_offsets(n_agents)

    for ii in range(n_agents):
        agent_state_traj = decoded_joint_states[state_start : state_end + 1, ii]
        coords = np.array([dyn_untimed.decode_state(int(state)) for state in agent_state_traj])
        offset_y, offset_x = agent_plot_offsets[ii]
        ys = coords[:, 0] + offset_y
        xs = coords[:, 1] + offset_x
        color = agent_colors[ii]

        if len(xs) > 1:
            ax_still.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, zorder=4)

        if draw_arrows:
            for step_idx in range(state_start, state_end):
                local_idx = step_idx - state_start
                start = np.array([xs[local_idx], ys[local_idx]], dtype=float)
                end = np.array([xs[local_idx + 1], ys[local_idx + 1]], dtype=float)
                delta = end - start
                if np.allclose(delta, 0.0):
                    continue

                node_color = node_cmap(int(node_ids[step_idx]))
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
        if len(xs) > 1:
            ax_still.plot(xs[-1], ys[-1], marker="s", color=color, ms=5, zorder=6)

    fig_still.savefig(fig_path, dpi=200, pad_inches=0)
    plt.close(fig_still)

@app.default()
def main(gamma: float | None = None, resolve: bool = False, resolve_nom: bool = False):
    RESULTS_DIR.mkdir(exist_ok=True)

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
        "site": np.where(get_mask(".") | d_raw["E"] | d_raw["F"], 1, -1)[:, :, -1],
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
    fig_path = RESULTS_DIR / "ma_timed_was_map.pdf"
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
        "r1": flat_totimed(rew_to_ma(d_flat["r1"], dyn_ma.n_agents, mode=0), t_max=TMAX), # ag 0 to A
        "r2": flat_totimed(rew_to_ma(d_flat["r2"], dyn_ma.n_agents, mode=0), t_max=TMAX), # ag 1 to B
        "r3": flat_totimed(rew_to_ma(d_flat["r3"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "saw": flat_totimed(rew_to_ma(d_flat["saw"], dyn_ma.n_agents, "min"), t_max=TMAX),
        "wood": flat_totimed(rew_to_ma(d_flat["wood"], dyn_ma.n_agents, "min"), t_max=TMAX),
        "gear": flat_totimed(rew_to_ma(d_flat["gear"], dyn_ma.n_agents, "min"), t_max=TMAX),
        "site": flat_totimed(rew_to_ma(d_flat["site"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "k": flat_totimed(rew_to_ma(d_flat["k"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "d": flat_totimed(rew_to_ma(d_flat["d"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "w": flat_totimed(rew_to_ma(d_flat["w"], dyn_ma.n_agents, "max"), t_max=TMAX),
        "collide": ma_collision_predicate(dyn_ma_, collide_dist, t_max=TMAX),
        "distant": flat_totimed(ma_distance_predicate(dyn_ma_, 2 * collide_dist), t_max=TMAX),
        "leash": ma_collision_predicate(dyn_ma_, 2, t_max=TMAX, norm=1),
        "TleTMAX": dyn_ma.tle_predicate(TMAX),
    }
    
    # Reach everything before TMAX.
    dict_predicates["r1"] = jnp.minimum(dict_predicates["r1"], dict_predicates["TleTMAX"])
    dict_predicates["r2"] = jnp.minimum(dict_predicates["r2"], dict_predicates["TleTMAX"])
    dict_predicates["gear"] = jnp.minimum(dict_predicates["gear"], dict_predicates["TleTMAX"])

    # Make the walls encode all the safety stuff for convenience.
    dict_predicates["w"] = jnp.stack([dict_predicates["w"], -dict_predicates["leash"], dict_predicates["collide"]], axis=-1).max(-1)
    
    for key, value in dict_predicates.items():
        assert value.ndim == 1 and value.shape[0] == dyn_ma.n_states, f"Predicate {key} has wrong shape {value.shape}"

    # Solve and save or load the solution.
    pkl_path = RESULTS_DIR / f"rooms_discrete_ma_timed_was_sol.pkl"
    dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras = _solve_and_cache(
        dyn_ma, dag_nodes, dag_root, dict_predicates, pkl_path, TASK_SOURCE, d_raw, gamma, resolve
    )

    # Make Agent Nominal Policies
    def solve_ag_policy(policy_task_source: str, ag_tag: str, gamma: float | None = None, resolve: bool = False):

        dyn, d_raw = parse_rooms(MAP)
        if any(key in d_raw for key in ("<", ">", "^", "v")):
            dyn.drift_fn = GridWorldDriftFn(d_raw, force=False)
        dyn_ma_ = GridWorldMA(dyn, n_agents=2)

        empty_mask = np.zeros_like(d_raw["#"], dtype=bool)
        get_mask = lambda key: d_raw.get(key, empty_mask)

        d = {
            "r1": np.where(get_mask("A"), 1, -1),
            "r2": np.where(get_mask("B"), 1, -1),
            "r3": np.where(get_mask("C"), 1, -1),
            "saw": np.where(get_mask("E"), 1, -1),
            "wood": np.where(get_mask("F"), 1, -1),
            "gear": np.where(get_mask("g"), 1, -1),
            "site": np.where(get_mask(".") | d_raw["E"] | d_raw["F"], 1, -1),
            "k": np.where(get_mask("K"), 1, -1),
            "d": np.where(get_mask("D"), 1, -1),
            "w": np.where(get_mask("#"), 1, -1),
            "<": np.where(get_mask("<"), 1, -1),
            ">": np.where(get_mask(">"), 1, -1),
            "^": np.where(get_mask("^"), 1, -1),
        }
        d_flat = {k: v.flatten() for k, v in d.items()}
        
        dict_predicates = {
            "r1": rew_to_ma(d_flat["r1"], dyn_ma_.n_agents, mode=0), # ag 0 to A
            "r2": rew_to_ma(d_flat["r2"], dyn_ma_.n_agents, mode=0), # ag 1 to B
            "r3": rew_to_ma(d_flat["r3"], dyn_ma_.n_agents, "max"),
            "saw": rew_to_ma(d_flat["saw"], dyn_ma_.n_agents, "min"),
            "wood": rew_to_ma(d_flat["wood"], dyn_ma_.n_agents, "min"),
            "gear": rew_to_ma(d_flat["gear"], dyn_ma_.n_agents, "min"),
            "site": rew_to_ma(d_flat["site"], dyn_ma_.n_agents, "max"),
            "k": rew_to_ma(d_flat["k"], dyn_ma_.n_agents, "max"),
            "d": rew_to_ma(d_flat["d"], dyn_ma_.n_agents, "max"),
            "w": rew_to_ma(d_flat["w"], dyn_ma_.n_agents, "max"),
            "collide": ma_collision_predicate(dyn_ma_, collide_dist),
            "distant": ma_distance_predicate(dyn_ma_, 2 * collide_dist),
            "leash": ma_collision_predicate(dyn_ma_, 2, norm=1),
            # "TleTMAX": dyn_ma_.tle_predicate(TMAX),
        }
        dict_predicates["w"] = jnp.stack([dict_predicates["w"], dict_predicates["collide"]], axis=-1).max(-1)

        # Decompose
        value_tree_dag, dag_root = to_dag(
            policy_task_source, ir_filename=f"rooms_discrete_ma_ag{ag_tag}_ir", dag_filename=f"rooms_discrete_ma_ag{ag_tag}_dag"
        )
        dag_nodes = value_tree_dag.nodes

        # Solve.
        pkl_path = RESULTS_DIR / f"safety_filter_ma_ag{ag_tag}_sol.pkl"
        sol_dyn_ma_, sol_dag_nodes, sol_dag_root, sol_dict_vars, sol_dict_actions, sol_dict_GU_vars, sol_dict_GU_actions, extras = _solve_and_cache(
            dyn_ma_, dag_nodes, dag_root, dict_predicates, pkl_path, policy_task_source, d_raw, gamma, resolve
        )

        return MinTimePolicy(sol_dyn_ma_, sol_dag_nodes, sol_dag_root, sol_dict_vars, sol_dict_actions, sol_dict_GU_vars, sol_dict_GU_actions)
    
    pol_ag1 = solve_ag_policy(TASK_SOURCE_AG1, "0", gamma=gamma, resolve=resolve_nom)
    pol_ag2 = solve_ag_policy(TASK_SOURCE_AG2, "1", gamma=gamma, resolve=resolve_nom)
    safety_filter = SafetyFilter(dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)

    # ---------------------------------------------------------------------
    # Get start state

    rng = np.random.default_rng(seed=12345)
    T_value = np.asarray(dict_vars[dag_root])
    value = T_value.reshape((dyn_ma_.n_states, TMAX + 1))[:, 0]

    if "s" in d_raw and np.any(d_raw["s"][:, :, -1]):
        fixed_positions = [tuple(pos) for pos in np.argwhere(d_raw["s"][:, :, -1])]
    else:
        fixed_positions = [(4, 4), (11, 11)]

    # ---------------------------------------------------------------------
    # Make Value plots for starting position

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
    fig_values.savefig(RESULTS_DIR / "rooms_discrete_ma_timed_was_root_values.png", dpi=200, bbox_inches="tight")
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

    # ---------------------------------------------------------------------
    # Rollout safety filter.
    state = start_state
    Tp1_states = [state]
    T_a_nom = []
    T_a_filt = []
    T_hasfiltered = []
    T_curnode_idxs = []

    # State tracking for fancy preference (works but jank)
    # recent_joint_states = deque(maxlen=8)
    # joint_visit_counts = Counter()
    # start_joint_state, _ = dyn_ma.decode_timed_state(state, which=np)
    # start_joint_state = int(start_joint_state)
    # recent_joint_states.append(start_joint_state)
    # joint_visit_counts[start_joint_state] += 1

    # def _joint_xy_from_joint_state(s_joint_: int) -> np.ndarray:
    #     agent_states_ = dyn_ma_.decode_joint_state(s_joint_, which=np)
    #     return np.array(
    #         [dyn_ma_.base.decode_state(int(agent_state), which=np) for agent_state in agent_states_],
    #         dtype=np.int32,
    #     )

    # def preference_fn(state_: StateInt, a_nom_: ActionInt) -> np.ndarray:
    #     s_joint_, _ = dyn_ma.decode_timed_state(state_, which=np)
    #     s_joint_ = int(s_joint_)
    #     cur_xy = _joint_xy_from_joint_state(s_joint_)

    #     s_nom_next = int(dyn_ma.step(state_, a_nom_, which=np))
    #     s_nom_next_joint, _ = dyn_ma.decode_timed_state(s_nom_next, which=np)
    #     s_nom_next_joint = int(s_nom_next_joint)
    #     nom_xy = _joint_xy_from_joint_state(s_nom_next_joint)
    #     nom_disp = nom_xy - cur_xy

    #     costs = np.full(dyn_ma.n_actions, np.inf, dtype=np.float32)
    #     for a in range(dyn_ma.n_actions):
    #         s_next = int(dyn_ma.step(state_, a, which=np))
    #         value_next = float(dict_vars[safety_filter.cur_node_id][s_next])
    #         if value_next < 0:
    #             continue

    #         s_next_joint, _ = dyn_ma.decode_timed_state(s_next, which=np)
    #         s_next_joint = int(s_next_joint)
    #         next_xy = _joint_xy_from_joint_state(s_next_joint)
    #         disp = next_xy - cur_xy

    #         align = float(np.sum(disp * nom_disp))
    #         dev = float(np.abs(next_xy - nom_xy).sum())
    #         stuck = float(s_next_joint == s_joint_)
    #         waiting_agents = float(np.sum(np.all(disp == 0, axis=1)))
    #         revisit = float(joint_visit_counts[s_next_joint] + 2 * (s_next_joint in recent_joint_states))

    #         costs[a] = (
    #             10.0 * stuck
    #             + 3.0 * waiting_agents
    #             + 2.0 * revisit
    #             + 1.0 * dev
    #             - 1.5 * align
    #             - 0.1 * value_next
    #         )

    #     return costs

    for kk in range(80):
        logger.debug(f"> kk={kk}")

        s_joint, _ = dyn_ma.decode_timed_state(state, which=np)
        state_ag1, state_ag2 = dyn_ma_.decode_joint_state(s_joint, which=np)
        state_ag1_tup = [int(n) for n in dyn_ma_.base.decode_state(state_ag1)]
        state_ag2_tup = [int(n) for n in dyn_ma_.base.decode_state(state_ag2)]
        logger.debug("    Current state: agent1 at {}, agent2 at {}".format(state_ag1_tup, state_ag2_tup))

        logger.debug("    Agent 1 policy...")
        a_nom_ag1_joint, ag1_isdone = pol_ag1.get_action(s_joint, which=np, kk=kk, debug=True)
        logger.debug("    Agent 1 policy... done!")
        if ag1_isdone:
            logger.debug("    Agent 1 policy is done. Using no-op.")
            a_nom_ag1 = dyn_untimed.str_to_action(".")
        else:
            a_nom_ag1 = dyn_ma_.decode_joint_action(a_nom_ag1_joint, which=np)[0]

        logger.debug("    Agent 2 policy...")
        a_nom_ag2_joint, ag2_isdone = pol_ag2.get_action(s_joint)
        logger.debug("    Agent 2 policy... done! {}".format(dyn_ma_.action_to_str(a_nom_ag2_joint)))
        a_nom_ag2 = dyn_ma_.decode_joint_action(a_nom_ag2_joint, which=np)[1]
        # if ag2_isdone:
        #     logger.debug("    Agent 2 policy is done! Using no-op.")
        #     a_nom_ag2 = dyn_untimed.str_to_action(".")
        # else:
        #     a_nom_ag2 = dyn_ma_.decode_joint_action(a_nom_ag2_joint, which=np)[1]

        a_nom = dyn_ma_.encode_joint_action([a_nom_ag1, a_nom_ag2], which=np)

        if a_nom_ag1 == dyn_untimed.str_to_action(".") and a_nom_ag2 == dyn_untimed.str_to_action("."):
            logger.debug("Both agents want to stay still!")
            a_nom = a_nom_ag2_joint

        # a_safe = safety_filter.filter_action(state, a_nom, preference_fn) # preference not helping here
        a_safe = safety_filter.filter_action(state, a_nom)
        hasfiltered = a_safe != a_nom
        state = dyn_ma.step(state, a_safe)

        T_a_nom.append(a_nom)
        T_a_filt.append(a_safe)
        Tp1_states.append(state)
        T_hasfiltered.append(hasfiltered)
        T_curnode_idxs.append(safety_filter.cur_node_id)

        # s_joint_new, _ = dyn_ma.decode_timed_state(state, which=np)
        # s_joint_new = int(s_joint_new)
        # recent_joint_states.append(s_joint_new)
        # joint_visit_counts[s_joint_new] += 1

    T_hasfiltered = np.array(T_hasfiltered)

    # ---------------------------------------------------------------------------
    # Rollout optimal policy for spec (no filtering)

    rollouter = MinTimeRollout(dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
    Tp1_states, T_actions, T_curnode_idxs = rollouter.rollout(start_state, max_steps=TMAX)

    # ---------------------------------------------------------------------

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
    anim.save(str(RESULTS_DIR / "rooms_discrete_rollout_multiagent_timed_was.mp4"), fps=5, dpi=200)
    plt.close(fig_anim)

    decoded_joint_states = decode_timed_joint_states(Tp1_states, dyn_ma, dyn_ma_)
    segment_ranges = compute_plot_segments(T_curnode_idxs, max_segment_plots=4)

    save_rollout_still(
        RESULTS_DIR / "rooms_discrete_rollout_multiagent_timed_was_t0.png",
        decoded_joint_states,
        state_start=0,
        state_end=0,
        node_ids=T_curnode_idxs,
        draw_arrows=False,
        fig_size=(4, 4),
        empty_map=empty_map,
        cmap=cmap,
        d_raw_viz=d_raw_viz,
        w=w,
        h=h,
        dyn_untimed=dyn_untimed,
        agent_colors=agent_colors,
    )

    for plot_idx, (action_start, action_end) in enumerate(segment_ranges, start=1):
        save_rollout_still(
            RESULTS_DIR / f"rooms_discrete_rollout_multiagent_timed_was_switch_{plot_idx}.png",
            decoded_joint_states,
            state_start=action_start,
            state_end=action_end,
            node_ids=T_curnode_idxs,
            draw_arrows=True,
            fig_size=(4, 4),
            empty_map=empty_map,
            cmap=cmap,
            d_raw_viz=d_raw_viz,
            w=w,
            h=h,
            dyn_untimed=dyn_untimed,
            agent_colors=agent_colors,
        )


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
