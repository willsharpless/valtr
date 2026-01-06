import copy
import os
import time

import cyclopts
import hj_reachability as hj
import hj_reachability.dynamics as dynamics
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from dvi.dynamics.gridworld import GridWorld
from dvi.gen_solver import (FixedPointGUSolver, avoid_update_rule, avoid_update_rule_with_actions, make_solve_fn,
                            make_solve_fn_with_actions, reach_avoid_update_rule, reach_avoid_update_rule_with_actions)
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import CenteredNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate as ode

from valtr.control import (construct_optimal_path, construct_optimal_path_batch, construct_optimal_path_batch_auto,
                           construct_optimal_path_batch_fast, plot_optimal_path)
from valtr.dag_graphviz import visualize_dag
from valtr.dag_passes import PassFoldConstBool, PassKeepReachable, PassRAToR
from valtr.ir_builder import IRBuilder
from valtr.ir_graphviz import visualize_ir
from valtr.ir_pass import PassCombineGloballySegments, PassFinallyToUntil
from valtr.lowering import Lowerer
from valtr.mintime_rollout import MinTimeRollout
from valtr.reachability import (DAGGU, DAGAvoid, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGReachAvoid, DAGVar, dag_to_str,
                                lower_ir_to_dag)
from valtr.solver_utils import solve_dag_values
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser
from valtr.util.jax_util import rep_vmap

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
#^^^ K#
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

MAP_NUM = 3


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
        def drift_fn(state: jnp.ndarray, delta: jnp.ndarray):
            # If state is on <, then only allow left movement.
            y, x = state
            left_only = jnp.array(d_raw["<"])[y, x]  # bool
            right_only = jnp.array(d_raw[">"])[y, x]  # bool

            up_only = jnp.array(d_raw["^"])[y, x]  # bool

            delta_x = jnp.where(left_only, -1, jnp.where(right_only, 1, delta[0]))
            delta_y = jnp.where(up_only, -1, delta[1])

            # delta_y = delta[1]
            # delta_y = jnp.where((x == 2) & (y == 1), -1, delta_y)
            delta = delta.at[1].set(delta_x).at[0].set(delta_y)

            return state + delta

        dyn.drift_fn = drift_fn

        return dyn, d
    elif MAP_NUM == 3:
        dyn, d_raw = parse_rooms(MAP3)
        d = {
            "r": np.where(d_raw["A"], 1, -1),
            "w": np.where(d_raw["#"], 1, -1),
        }
        _, d_raw_drift = parse_rooms(MAP3_DRIFT)

        # Modify the drift.
        def drift_fn(state: jnp.ndarray, delta: jnp.ndarray):
            # If state is on <, then only allow left movement.
            y, x = state
            l_only = jnp.array(d_raw_drift["<"])[y, x]  # bool
            r_only = jnp.array(d_raw_drift[">"])[y, x]  # bool
            u_only = jnp.array(d_raw_drift["^"])[y, x]  # bool
            d_only = jnp.array(d_raw_drift["v"])[y, x]  # bool

            delta_x = jnp.where(l_only, -1, jnp.where(r_only, 1, delta[0]))
            delta_y = jnp.where(u_only, -1, jnp.where(d_only, 1, delta[1]))

            # In l_only or r_only, delta_y = 0. Similarly, in u_only or d_only, delta_x = 0.
            delta_y = jnp.where(l_only | r_only, 0, delta_y)
            delta_x = jnp.where(u_only | d_only, 0, delta_x)

            delta = delta.at[1].set(delta_x).at[0].set(delta_y)

            return state + delta

        dyn.drift_fn = drift_fn
        return dyn, d

    else:
        raise NotImplementedError("")


def parse_rooms(s: str):
    s = s.strip()

    # Figure out how many rows and columns.
    lines = s.split("\n")
    height = len(lines)
    width = len(lines[0])

    # For each unique character, create an entry in the dict.
    d = {}
    for ii, l in enumerate(lines):
        assert len(l) == width

        for jj, c in enumerate(l):
            if c not in d:
                d[c] = np.zeros((height, width), dtype=bool)

            d[c][ii, jj] = True

    shape = (height, width)
    drift_fn = None
    dyn = GridWorld(shape, drift_fn)

    return dyn, d


# def frombool(arr: np.ndarray, true_val, false_val):
#     return np.where(arr, true_val, false_val)


@app.default()
def main(view_pdf: bool = False):
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
        TASK_SOURCE = "G F r1 && G F r2 && G( !w )"
        # TASK_SOURCE = "G F r1 && G F r2 && (!d U k) && G( !w )"
        # TASK_SOURCE = "G( !w )"
    elif MAP_NUM == 3:
        TASK_SOURCE = "G F r && G( !w )"
    else:
        raise ValueError("Invalid MAP_NUM")

    # -------------------------------------------------------------------------------------------
    # Parse and lower the task specification to a value tree DAG.
    logger.info("Generating the value tree DAG from logic...")
    print(f"Input task logic: {TASK_SOURCE}")

    lexer = TLLexer()
    tokens = list(lexer.tokenize(TASK_SOURCE))
    ast = TLParser(tokens).parse()

    # AST -> IR
    ir = IRBuilder()
    lowerer = Lowerer(builder=ir)
    ir_root_id = lowerer.lower(ast)

    passes = [PassFinallyToUntil, PassCombineGloballySegments]
    for p_cls in passes:
        p = p_cls(ir)
        ir_root_id, ir = p.run(ir_root_id)

    dot_ir = visualize_ir(ir, ir_root_id, filename="ir_graph", view=view_pdf)

    # IR -> DAG
    value_tree_dag, dag_root = lower_ir_to_dag(ir, ir_root_id)

    n_changes = 0
    visualize_dag(value_tree_dag, dag_root, filename="rooms_discrete_dag0", view=view_pdf)

    # Perform constant folding.
    passes = [PassFoldConstBool]
    for p_cls in passes:
        changed = True
        while changed:
            p = p_cls(value_tree_dag)
            dag_root, value_tree_dag, changed = p.run(dag_root)
            n_changes += int(changed)
            if changed:
                visualize_dag(value_tree_dag, dag_root, filename=f"rooms_discrete_dag{n_changes}", view=view_pdf)

    # Visualize the DAG.
    visualize_dag(value_tree_dag, dag_root, filename="rooms_discrete_dag", view=view_pdf)

    dyn: GridWorld
    dyn, dict_predicates_unflat = get_rooms()
    dict_predicates = {k: v.flatten() for k, v in dict_predicates_unflat.items()}

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
        map_str = MAP1 if MAP_NUM == 1 else MAP2 if MAP_NUM == 2 else MAP3
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

    # Solve.
    dict_vars = {}
    dict_actions = {}
    dict_GU_vars = {}
    dict_GU_actions = {}
    dict_locals = dict_predicates
    for dag_id, node in enumerate(tqdm.tqdm(value_tree_dag.nodes)):
        match node:
            case DAGVar(name=name):
                assert name in dict_locals, "Unknown variable name {}".format(name)
                dict_vars[dag_id] = dict_locals[name]

            case DAGNegate(arg=arg):
                val = dict_vars[arg]
                dict_vars[dag_id] = -val

            case DAGMinN(args=args):
                args = np.stack([dict_vars[a] for a in args], axis=0)
                val = np.min(args, axis=0)
                dict_vars[dag_id] = val

            case DAGMaxN(args=args):
                args = np.stack([dict_vars[a] for a in args], axis=0)
                val = np.max(args, axis=0)
                dict_vars[dag_id] = val

            case DAGReachAvoid(reach=reach, avoid=avoid):
                # Note: the avoid is a stay since we are maximizing the value.
                arg_reach = dict_vars[reach]
                arg_avoid = dict_vars[avoid]

                s_v0 = arg_reach
                kwargs = dict(s_r=arg_reach, s_q=arg_avoid)

                # update_rule = reach_avoid_update_rule
                # solve_fn = make_solve_fn(dyn, update_rule, n_updates=dyn.n_states)
                update_rule = reach_avoid_update_rule_with_actions
                solve_fn = make_solve_fn_with_actions(dyn, update_rule, n_updates=dyn.n_states)
                dict_vars[dag_id], dict_actions[dag_id] = solve_fn(s_v0, **kwargs)

            case DAGAvoid(avoid=avoid):
                # Note: the avoid is a stay since we are maximizing the value.
                arg_avoid = dict_vars[avoid]

                s_v0 = arg_avoid
                kwargs = dict(s_q=arg_avoid)

                # update_rule = avoid_update_rule
                # solve_fn = make_solve_fn(dyn, update_rule, n_updates=dyn.n_states)
                update_rule = avoid_update_rule_with_actions
                solve_fn = make_solve_fn_with_actions(dyn, update_rule, n_updates=dyn.n_states)
                dict_vars[dag_id], dict_actions[dag_id] = solve_fn(s_v0, **kwargs)

            case DAGGU(args=args):
                U_args = [[dict_vars[q], dict_vars[r]] for q, r in args]
                out = FixedPointGUSolver().solve(dyn, U_args, n_iters=3)
                dict_vars[dag_id], dict_actions[dag_id], dict_GU_vars[dag_id], dict_GU_actions[dag_id] = out

        # fig, ax = plt.subplots()
        # im = ax.imshow(dict_vars[dag_id].reshape(dyn.shape), vmin=-1, vmax=1)
        #
        # ax.set_xticks(np.arange(w + 1) - 0.5)
        # ax.set_yticks(np.arange(h + 1) - 0.5)
        # cbar = fig.colorbar(im, ax=ax)
        # ax.set_title("{} ({})".format(dag_id, node))
        # plt.show()

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

    # ---------------------------------
    # Rollout from a feasible start state.
    value = dict_vars[dag_root]
    feasible_states = np.where(value > 0)[0]

    rng = np.random.default_rng(seed=12345)
    start_state = rng.choice(feasible_states)

    rollouter = MinTimeRollout(dyn, value_tree_dag, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
    Tp1_states, T_actions = rollouter.rollout(start_state, max_steps=50)

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
