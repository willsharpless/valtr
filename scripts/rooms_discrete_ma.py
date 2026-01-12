import cyclopts
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from dvi.dynamics.gridworld import GridWorld
from dvi.dynamics.gridworld_ma import GridWorldMA, ma_collision_predicate, rew_to_ma
from dvi.gen_solver import (FixedPointGUSolver, avoid_update_rule_with_actions, make_solve_fn_with_actions,
                            reach_avoid_update_rule_with_actions)
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
from valtr.reachability import DAGGU, DAGAvoid, DAGMaxN, DAGMinN, DAGNegate, DAGReachAvoid, DAGVar, lower_ir_to_dag
from valtr.solve_discrete import solve_discrete
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser
from valtr.valtr import to_dag

app = cyclopts.App()

# MAP = """
# #######
# #^K ^^#
# #A<  B#
# ###D###
# #A  >B#
# # C  ^#
# #######
# """

# MAP = """
# #######
# #^K ^^#
# #v   B#
# #A<   #
# ###D###
# #A  >B#
# #    ^#
# # C   #
# #######
# """

MAP = """
###############
# K           #
#           BB#
#           BB#
#vv   ####    #
#AA<  ####    #
#AA<          #
######DDD######
#AA        >BB#
#AA        >BB#
#   ##      ^^#
#   ##        #
#     CC      #
#     CC      #
###############
"""

TASK_SOURCE = "G F r1 && G F r2 && G F r3 && (!d U k) && G(!w) && G(!collide)"
# TASK_SOURCE = "G(!w) && G(!collide)"


@app.default()
def main(view_pdf: bool = False, gamma: float | None = None):
    dyn, d_raw = parse_rooms(MAP)
    d = {
        "r1": np.where(d_raw["A"], 1, -1),
        "r2": np.where(d_raw["B"], 1, -1),
        "r3": np.where(d_raw["C"], 1, -1),
        "k": np.where(d_raw["K"], 1, -1),
        "d": np.where(d_raw["D"], 1, -1),
        "w": np.where(d_raw["#"], 1, -1),
        # Just for convenience.
        "<": np.where(d_raw["<"], 1, -1),
        ">": np.where(d_raw[">"], 1, -1),
        "^": np.where(d_raw["^"], 1, -1),
    }
    dyn.drift_fn = get_drift_fn(d_raw, force=False)
    # ------------------------------

    h, w = dyn.shape
    empty_map = np.zeros_like(d_raw["#"])
    for ii, (k, v) in enumerate(d_raw.items()):
        empty_map = np.where(v, ii, empty_map)
    tick_locs = np.arange(len(d_raw)) + 0.5
    cmap = plt.get_cmap("tab20", len(d_raw))
    colors = cmap.colors

    # Get the index of " " in the keys to set it to white.
    if " " in d_raw:
        space_idx = list(d_raw.keys()).index(" ")
        colors[space_idx] = np.array([1.0, 1.0, 1.0, 1.0])

    cmap = ListedColormap(colors)

    # ------------------------------
    dyn_ma = GridWorldMA(dyn, n_agents=2)
    # dyn_ma = GridWorldMA(dyn, n_agents=3)

    value_tree_dag, dag_root = to_dag(
        TASK_SOURCE, ir_filename="rooms_discrete_ma_ir", dag_filename="rooms_discrete_ma_dag"
    )

    dict_predicates_unflat = d
    dict_predicates = {k: v.flatten() for k, v in dict_predicates_unflat.items()}

    # Convert from single-agent to multi-agent predicates.
    collide_dist = 1.0  # Diagonal is safe, but not adjacent.
    dict_predicates = {
        "r1": rew_to_ma(dict_predicates["r1"], dyn_ma.n_agents, "max"),
        "r2": rew_to_ma(dict_predicates["r2"], dyn_ma.n_agents, "max"),
        "r3": rew_to_ma(dict_predicates["r3"], dyn_ma.n_agents, "max"),
        "k": rew_to_ma(dict_predicates["k"], dyn_ma.n_agents, "max"),
        "d": rew_to_ma(dict_predicates["d"], dyn_ma.n_agents, "max"),
        "w": rew_to_ma(dict_predicates["w"], dyn_ma.n_agents, "max"),
        "collide": ma_collision_predicate(dyn_ma, collide_dist),
    }

    # -------------------------------------------
    # Solve.
    dict_vars, dict_actions, dict_GU_vars, dict_GU_actions = solve_discrete(
        dyn_ma, value_tree_dag, dict_predicates, gamma=gamma
    )

    # ---------------------------------
    rng = np.random.default_rng(seed=12345)

    value = dict_vars[dag_root]

    feasible_states = np.where(value >= 0)[0]
    logger.info("Num feasible states: {}".format(len(feasible_states)))

    is_good = (dict_predicates["k"] != 1) & (value >= 0)
    feasible_states_good = np.where(is_good)[0]
    logger.info("Num feasible states (where not on key): {}".format(len(feasible_states_good)))

    # If possible, choose an initial state where none of the agents start on the key.
    # start_state = dyn_ma.encode_from_tups([(3, 3), (3, 5), (7, 5)])
    start_state = dyn_ma.encode_from_tups([(4, 9), (6, 6)])
    if start_state is not None and value[start_state] >= 0:
        logger.info("Using hardcoded start state.")
    else:
        logger.info("Hardcoded start state not feasible: {}".format(value[start_state]))
        if len(feasible_states_good) > 0:
            start_state = rng.choice(feasible_states_good)
        else:
            start_state = rng.choice(feasible_states)

    # ---------------------------------
    rollouter = MinTimeRollout(dyn_ma, value_tree_dag, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
    Tp1_states, T_actions = rollouter.rollout(start_state, max_steps=50)

    # Visualize the rollout by animating the path and saving as mp4.
    n_frames = len(Tp1_states)
    fig, ax = plt.subplots()

    # Visualize the map again.
    im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw), alpha=0.5)
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))
    ax.set_xticks(np.arange(w + 1) - 0.5)
    ax.set_yticks(np.arange(h + 1) - 0.5)

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
        agent_states = dyn_ma.decode_joint_state(joint_state)

        # Update each agent dot
        for i in range(n_agents):
            s_i = int(agent_states[i])  # safe for matplotlib
            y, x = dyn_ma.base.decode_state(s_i)  # (y, x) for your gridworld
            agent_dots[i].set_data([x], [y])

        kk_text.set_text(f"Step {kk: 3}")

        return agent_dots + [kk_text]

    anim = FuncAnimation(fig, update_fn, n_frames, init_fn, blit=True)
    anim.save("rooms_discrete_rollout_multiagent.mp4", fps=5, dpi=200)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
