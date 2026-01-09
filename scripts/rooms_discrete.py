import cyclopts
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.gridworld import GridWorld
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from valtr.dag_graphviz import visualize_dag
from valtr.dag_passes import PassFoldConstBool
from valtr.gridworld_utils import get_drift_fn, parse_rooms
from valtr.ir_builder import IRBuilder
from valtr.ir_graphviz import visualize_ir
from valtr.ir_pass import PassCombineGloballySegments, PassFinallyToUntil
from valtr.lowering import Lowerer
from valtr.mintime_rollout import MinTimeRollout
from valtr.reachability import lower_ir_to_dag
from valtr.solve_discrete import solve_discrete
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser
from valtr.valtr import to_dag

app = cyclopts.App()

MAP1 = """
#############
#C  2 A 1   #
#   2   1   #
#   2   1 B #
#############
"""

MAP2 = """
#######
#^K ^^#
#A<  B#
###D###
#A  >B#
#######
"""

MAP3 = """
#####
#A  #
##  #
##  #
#A  #
#   #
#####
"""

MAP3_DRIFT = """
#####
#<  #
## ^#
## ^#
#v ^#
#>>^#
#####
"""

MAP4 = """
###########
#BA     A #
#    S    #
###########
"""

MAP_NUM = 4


def get_rooms():
    if MAP_NUM == 1:
        s = MAP1
        dyn, d_raw = parse_rooms(s)
        d = {
            "k1": np.where(d_raw["A"], 1, -1),
            "d1": np.where(d_raw["1"], 1, -1),
            "k2": np.where(d_raw["B"], 1, -1),
            "d2": np.where(d_raw["2"], 1, -1),
            "k3": np.where(d_raw["C"], 1, -1),
            "w": np.where(d_raw["#"], 1, -1),
        }
        return dyn, d
    elif MAP_NUM == 2:
        s = MAP2
        dyn, d_raw = parse_rooms(s)
        d = {
            "r1": np.where(d_raw["A"], 1, -1),
            "r2": np.where(d_raw["B"], 1, -1),
            "k": np.where(d_raw["K"], 1, -1),
            "d": np.where(d_raw["D"], 1, -1),
            "w": np.where(d_raw["#"], 1, -1),
            # Just for convenience.
            "<": np.where(d_raw["<"], 1, -1),
            ">": np.where(d_raw[">"], 1, -1),
            "^": np.where(d_raw["^"], 1, -1),
        }

        # Modify the drift. On <, can only go left. On >, can only go right.
        dyn.drift_fn = get_drift_fn(d_raw, force=False)

        return dyn, d
    elif MAP_NUM == 3:
        dyn, d_raw = parse_rooms(MAP3)
        d = {
            "r": np.where(d_raw["A"], 1, -1),
            "w": np.where(d_raw["#"], 1, -1),
        }
        _, d_raw_drift = parse_rooms(MAP3_DRIFT)

        # Modify the drift.
        dyn.drift_fn = get_drift_fn(d_raw_drift, force=True)
        return dyn, d
    elif MAP_NUM == 4:
        dyn, d_raw = parse_rooms(MAP4)
        d = {
            "A": np.where(d_raw["A"], 1, -1),
            "B": np.where(d_raw["B"], 1, -1),
            "S": np.where(d_raw["S"], 1, -1),
            "w": np.where(d_raw["#"], 1, -1),
        }
        return dyn, d
    else:
        raise NotImplementedError("")


# def frombool(arr: np.ndarray, true_val, false_val):
#     return np.where(arr, true_val, false_val)


@app.default()
def main(view_pdf: bool = False, gamma: float | None = None):
    if MAP_NUM == 1:
        # MAP1
        # TASK_SOURCE = "!d1 U k1"
        # TASK_SOURCE = "(!d1 U k1) && G( !w )"
        # TASK_SOURCE = "(!d1 U k1) && (!d2 U k2) && G( !w )"
        TASK_SOURCE = "(!d1 U k1) && (!d2 U k2) && F k3 && G( !w )"
        # TASK_SOURCE = "(!d1 U k1) && F k3 && G( !w )"
        # TASK_SOURCE = "(!d1 U k1) && G(!d2 U k2) && G(F k3) && G( !w )"
    elif MAP_NUM == 2:
        # MAP2
        # TASK_SOURCE = "G F r1 && G F r2 && G( !w )"
        TASK_SOURCE = "G F r1 && G F r2 && (!d U k) && G( !w )"
        # TASK_SOURCE = "G( !w )"
    elif MAP_NUM == 3:
        TASK_SOURCE = "G F r && G( !w )"
    elif MAP_NUM == 4:
        TASK_SOURCE = "F A && F B && G( !w )"
    else:
        raise ValueError("Invalid MAP_NUM")

    # -------------------------------------------------------------------------------------------
    # Parse and lower the task specification to a value tree DAG.
    logger.info("Generating the value tree DAG from logic...")
    print(f"Input task logic: {TASK_SOURCE}")

    value_tree_dag, dag_root = to_dag(TASK_SOURCE, ir_filename="rooms_discrete_ir", dag_filename="rooms_discrete_dag")

    dyn: GridWorld
    dyn, dict_predicates_unflat = get_rooms()
    dict_predicates = {k: v.flatten() for k, v in dict_predicates_unflat.items()}
    # logger.info("dyn.shape: {}".format(dyn.shape))

    # n_col = len(dict_predicates)
    # figsize = np.array([n_col * 3, 3])
    # fig, axes = plt.subplots(nrows=1, ncols=n_col, figsize=figsize)
    # for ii, (k, val) in enumerate(dict_predicates.items()):
    #     ax = axes[ii]
    #     ax.imshow(val, vmin=-1, vmax=1)
    #     ax.set_title(k)
    # plt.show()

    h, w = dyn.shape

    # Visualize the map.
    # Use a different color for each symbol.

    if MAP_NUM == 3:
        d_raw = parse_rooms(MAP3)[1]
        d_raw_drift = parse_rooms(MAP3_DRIFT)[1]
        d_raw = d_raw_drift | d_raw
    else:
        map_str = [None, MAP1, MAP2, MAP3, MAP4][MAP_NUM]
        _, d_raw = parse_rooms(map_str)

    empty_map = np.zeros_like(d_raw["#"])
    for ii, (k, v) in enumerate(d_raw.items()):
        empty_map = np.where(v, ii, empty_map)

    tick_locs = np.arange(len(d_raw)) + 0.5

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab20", len(d_raw))
    colors = cmap.colors

    # Get the index of " " in the keys to set it to white.
    if " " in d_raw:
        space_idx = list(d_raw.keys()).index(" ")
        colors[space_idx] = np.array([1.0, 1.0, 1.0, 1.0])

    cmap = ListedColormap(colors)

    im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw))
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))
    ax.set_title("Map visualization")
    ax.set_xticks(np.arange(w + 1) - 0.5)
    ax.set_yticks(np.arange(h + 1) - 0.5)
    fig.savefig("rooms_discrete.pdf")
    plt.close(fig)

    # # --------------------------
    # start_state = dyn.encode_state((2, 3))
    #
    # fig, ax = plt.subplots()
    # im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw))
    # cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    # cbar.ax.set_yticklabels(list(d_raw.keys()))
    # for action in range(dyn.n_actions):
    #     state_new = dyn.step(start_state, action)
    #     y, x = dyn.decode_state(state_new)
    #     ax.plot(x, y, marker="o", label=f"Action {action}")
    # ax.legend()
    # plt.show()
    # exit(0)

    # -------------------------------------------
    # Solve.
    dict_vars, dict_actions, dict_GU_vars, dict_GU_actions = solve_discrete(
        dyn, value_tree_dag, dict_predicates, gamma=gamma
    )

    # ---------------------------------
    # Visualize the final value function.
    ncol = 2
    figsize = np.array([ncol * 6, 4])
    fig, axes = plt.subplots(1, ncol, figsize=figsize)
    #     map
    ax = axes[0]
    im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw))
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))
    ax.set_xticks(np.arange(w + 1) - 0.5)
    ax.set_yticks(np.arange(h + 1) - 0.5)

    #     value
    ax = axes[1]
    im = ax.imshow(dict_vars[dag_root].reshape(dyn.shape), vmin=-1, vmax=1)
    ax.set_xticks(np.arange(w + 1) - 0.5)
    ax.set_yticks(np.arange(h + 1) - 0.5)
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title("Final value function")
    fig.savefig("rooms_discrete_value.pdf")
    plt.close(fig)

    # # -------------------------------------------
    # # Visualize the GU actions.
    # node_idx = list(dict_GU_actions.keys())[0]
    # GU_actions = dict_GU_actions[node_idx]
    #
    # cmap = plt.get_cmap("tab10", dyn.n_actions)
    #
    # ncol = len(GU_actions)
    # figsize = np.array([ncol * 6, 4])
    # fig, axes = plt.subplots(1, ncol + 1, figsize=figsize, layout="constrained")
    #
    # ax = axes[0]
    # im = ax.imshow(dict_vars[dag_root].reshape(dyn.shape), vmin=-1, vmax=1)
    # ax.set_xticks(np.arange(w + 1) - 0.5)
    # ax.set_yticks(np.arange(h + 1) - 0.5)
    # cbar = fig.colorbar(im, ax=ax)
    #
    # arrow_map = {0: "→", 1: "←", 2: "↑", 3: "↓", 4: "•"}
    #
    # for ii, ax in enumerate(axes[1:]):
    #     actions = GU_actions[ii].reshape(dyn.shape)
    #
    #     im = ax.imshow(actions, vmin=0, vmax=dyn.n_actions - 1, cmap=cmap)
    #     ax.set_xticks(np.arange(w + 1) - 0.5)
    #     ax.set_yticks(np.arange(h + 1) - 0.5)
    #     cbar = fig.colorbar(im, ax=ax, ticks=np.arange(dyn.n_actions))
    #     cbar.ax.set_yticklabels(["R", "L", "D", "U", "Stay"])
    #
    #     # Put text arrows.
    #     for y in range(h):
    #         for x in range(w):
    #             action = int(actions[y, x])
    #             ax.text(
    #                 x,
    #                 y,
    #                 arrow_map.get(action, str(action)),
    #                 color="black",
    #                 ha="center",
    #                 va="center",
    #                 fontsize=12,
    #                 fontweight="bold",
    #             )
    #
    #     ax.set_title(f"GU Action {ii}")
    #
    # fig.savefig("rooms_discrete_gu_actions.pdf")
    # plt.close(fig)

    # ---------------------------------
    # Rollout from a feasible start state.
    value = dict_vars[dag_root]
    feasible_states = np.where(value > 0)[0]

    if MAP_NUM == 4:
        start_state = dyn.encode_state((2, 5))
    else:
        rng = np.random.default_rng(seed=12345)
        start_state = rng.choice(feasible_states)
        # start_state = dyn.encode_state((2, 4))

    # Visualize the start state.
    y, x = dyn.decode_state(start_state)
    fig, ax = plt.subplots()
    im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw))
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))
    ax.set_xticks(np.arange(w + 1) - 0.5)
    ax.set_yticks(np.arange(h + 1) - 0.5)
    (state_dot,) = ax.plot([x], [y], marker="o", color="red", ms=5)
    ax.set_title("Start state")
    fig.savefig("rooms_discrete_start_state.pdf")
    plt.close(fig)

    # ----------------------------
    rollouter = MinTimeRollout(dyn, value_tree_dag, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
    Tp1_states, T_actions = rollouter.rollout(start_state, max_steps=30)

    # Visualize the rollout by animating the path and saving as mp4.
    n_frames = len(Tp1_states)
    fig, ax = plt.subplots()

    # Visualize the map again.
    im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw))
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))
    ax.set_xticks(np.arange(w + 1) - 0.5)
    ax.set_yticks(np.arange(h + 1) - 0.5)

    # Draw the state as a red dot.
    (state_dot,) = ax.plot([], [], marker="o", color="red", ms=5)

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
        return [state_dot, kk_text]

    def update_fn(kk: int) -> list[plt.Artist]:
        state = Tp1_states[kk]

        y, x = dyn.decode_state(state)
        state_dot.set_data([x], [y])

        kk_text.set_text(f"Step {kk: 3}")

        return [state_dot, kk_text]

    anim = FuncAnimation(fig, update_fn, n_frames, init_fn, blit=True)
    anim.save("rooms_discrete_rollout.mp4", fps=5, dpi=200)

    logger.info("Done!")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
