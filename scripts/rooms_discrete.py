import copy
import os
import time

import hj_reachability as hj
import hj_reachability.dynamics as dynamics
import ipdb
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from dvi.dynamics.gridworld import GridWorld
from dvi.gen_solver import avoid_update_rule, make_solve_fn, reach_avoid_update_rule, FixedPointGUSolver
from loguru import logger
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
from valtr.reachability import (DAGGU, DAGAvoid, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGReachAvoid, DAGVar, dag_to_str,
                                lower_ir_to_dag)
from valtr.solver_utils import solve_dag_values
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser
from valtr.util.jax_util import rep_vmap

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


# def get_rooms():
#     s = MAP1
#     dyn, d_raw = parse_rooms(s)
#     d = {
#         "k1": np.where(d_raw["A"], 1, -1),
#         "d1": np.where(d_raw["1"], 1, -1),
#         "k2": np.where(d_raw["B"], 1, -1),
#         "d2": np.where(d_raw["2"], 1, -1),
#         "k3": np.where(d_raw["C"], 1, -1),
#         "w": np.where(d_raw["#"], 1, -1),
#     }
#     return dyn, d
#
def get_rooms():
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


def main():
    # MAP1
    # TASK_SOURCE = "(!d1 U k1) && G( !w )"
    # TASK_SOURCE = "(!d1 U k1) && (!d2 U k2) && F k3 && G( !w )"
    # TASK_SOURCE = "(!d1 U k1) && F k3 && G( !w )"
    # TASK_SOURCE = "(!d1 U k1) && G(!d2 U k2) && G(F k3) && G( !w )"

    # MAP2
    # TASK_SOURCE = "G F r1 && G F r2 && (!d U k) && G( !w )"
    TASK_SOURCE = "G( !w )"

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

    # dot_ir = visualize_ir(ir, ir_root_id, filename="ir_graph", view=True)

    # IR -> DAG
    value_tree_dag, dag_root = lower_ir_to_dag(ir, ir_root_id)

    # dot_dag = visualize_dag(value_tree_dag, dag_root, filename="rooms_discrete_dag1", view=True)

    # Perform constant folding.
    passes = [PassFoldConstBool]
    for p_cls in passes:
        p = p_cls(value_tree_dag)
        # changed = True
        # while changed:
        dag_root, value_tree_dag, changed = p.run(dag_root)

    # # Visualize the DAG.
    # dot_dag = visualize_dag(value_tree_dag, dag_root, filename="rooms_discrete_dag", view=True)

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

    # # Visualize the map.
    # # Use a different color for each symbol.
    # _, d_raw = parse_rooms(MAP2)
    #
    # empty_map = np.zeros_like(d_raw["#"])
    # for ii, (k, v) in enumerate(d_raw.items()):
    #     empty_map = np.where(v, ii + 1, empty_map)
    #
    # fig, ax = plt.subplots()
    # cmap = plt.get_cmap("tab20", len(d_raw) + 1)
    # im = ax.imshow(empty_map, cmap=cmap)
    # cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(d_raw) + 1))
    # cbar.ax.set_yticklabels([""] + list(d_raw.keys()))
    # ax.set_title("Map visualization")
    # ax.set_xticks(np.arange(w + 1) - 0.5)
    # ax.set_yticks(np.arange(h + 1) - 0.5)
    # plt.show()

    # Solve.
    dict_vars = {}
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

                update_rule = reach_avoid_update_rule
                solve_fn = make_solve_fn(dyn, update_rule, n_updates=dyn.n_states)
                dict_vars[dag_id] = solve_fn(s_v0, **kwargs)

            case DAGAvoid(avoid=avoid):
                # Note: the avoid is a stay since we are maximizing the value.
                arg_avoid = dict_vars[avoid]

                s_v0 = arg_avoid
                kwargs = dict(s_q=arg_avoid)

                update_rule = avoid_update_rule
                solve_fn = make_solve_fn(dyn, update_rule, n_updates=dyn.n_states)
                dict_vars[dag_id] = solve_fn(s_v0, **kwargs)

            case DAGGU(args=args):
                U_args = [[dict_vars[q], dict_vars[r]] for q, r in args]

                solver = FixedPointGUSolver(U_args)
                dict_vars[dag_id] = solver.solve(n_iters=10)

        fig, ax = plt.subplots()
        im = ax.imshow(dict_vars[dag_id].reshape(dyn.shape), vmin=-1, vmax=1)

        ax.set_xticks(np.arange(w + 1) - 0.5)
        ax.set_yticks(np.arange(h + 1) - 0.5)
        cbar = fig.colorbar(im, ax=ax)
        ax.set_title("{} ({})".format(dag_id, node))
        plt.show()

    logger.info("Done!")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
