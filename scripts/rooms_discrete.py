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
from valtr.solve_discrete import load_discrete_sol, save_discrete_sol, solve_discrete
from valtr.valtr import to_dag

plt.style.use("seaborn-v0_8-darkgrid")

app = cyclopts.App()

MAP0 = """
#########
# A#    #
# ## ## #
#    #B #
#########
"""
# called 1 in vdppo code

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

MAP5 = """
   D    
 A #    
   #    
#### # B
     #  
 K ###  
   #    
   #    
"""

MAP6 = """
     A  
        
  ...b  
g ....  
  ....  
  ..a.  
        
    B   
"""

MAP7 = """
 A      
 .b  .b 
 ..  .. 
 ..  a.B
 ..g .. 
 ..  .. 
 a.  .. 
        
"""

MAP8 = """
 A      
 .b  .b 
 ..  .. 
 ..  a.B
 ..g .. 
 ..  .. 
 a.  .. 
       w
"""
# just for dag

MAP9 = """
A
"""

MAP_NUM = 9


def get_rooms():
    map_str = [None, MAP1, MAP2, MAP3, MAP4, MAP5, MAP6, MAP7, MAP8, MAP9][MAP_NUM]
    if MAP_NUM == 0:
        s = MAP0
        dyn, d_raw = parse_rooms(s)
        d = {
            "A": np.where(d_raw["A"], 1, -1),
            "B": np.where(d_raw["B"], 1, -1),
            "w": np.where(d_raw["#"], 1, -1),
        }
        return dyn, d
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
        dyn.drift_fn = GridWorldDriftFn(d_raw, force=False)

        return dyn, d
    elif MAP_NUM == 3:
        dyn, d_raw = parse_rooms(MAP3)
        d = {
            "r": np.where(d_raw["A"], 1, -1),
            "w": np.where(d_raw["#"], 1, -1),
        }
        _, d_raw_drift = parse_rooms(MAP3_DRIFT)

        # Modify the drift.
        dyn.drift_fn = GridWorldDriftFn(d_raw_drift, force=True)
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
    elif MAP_NUM == 5:
        dyn, d_raw = parse_rooms(MAP5)
        d = {
            "A": np.where(d_raw["A"], 1, -1),
            "B": np.where(d_raw["B"], 1, -1),
            "D": np.where(d_raw["D"], 1, -1),
            "K": np.where(d_raw["K"], 1, -1),
            "w": np.where(d_raw["#"], 1, -1),
        }
        return dyn, d
    elif MAP_NUM == 6 or MAP_NUM == 7 or MAP_NUM == 8:
        dyn, d_raw = parse_rooms(map_str)
        d = {
            "A": np.where(d_raw["a"] | d_raw["A"], 1, -1),
            "B": np.where(d_raw["b"] | d_raw["B"], 1, -1),
            "g": np.where(d_raw["g"], 1, -1),
            "q": np.where(d_raw["."] | d_raw["a"] | d_raw["b"], 1, -1),
        }
        return dyn, d
    elif MAP_NUM == 9:
        dyn, d_raw = parse_rooms(MAP9)
        d = {
            "A": np.where(d_raw["A"], 1, -1),
            "B": np.where(d_raw["A"], 1, -1),
        }
        return dyn, d
    else:
        raise NotImplementedError("")


# def frombool(arr: np.ndarray, true_val, false_val):
#     return np.where(arr, true_val, false_val)


@app.default()
def main(view_pdf: bool = False, room: int = 1, gamma: float | None = None, resolve: bool = False):
    global MAP_NUM
    MAP_NUM = room

    results_dir = pathlib.Path("plots_discrete")
    results_dir.mkdir(exist_ok=True)

    if MAP_NUM == 0:
        TASK_SOURCE = "F A && F B && G( !w )"
    elif MAP_NUM == 1:
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
        # TASK_SOURCE = "F A && F B && G( !w )"
        TASK_SOURCE = "G F A && G F B && G( !w )"
    elif MAP_NUM == 5:
        TASK_SOURCE = "F A && F B && !D U K && G( !w )"
    elif MAP_NUM == 6:
        TASK_SOURCE = "F( g && F G ( (q U (A && q )) && (q U (B && q )) ) )"
    elif MAP_NUM == 7:
        TASK_SOURCE = "(!q U g) && F( g && F G ( (q U (A && q )) && (q U (B && q )) ) )"
    elif MAP_NUM == 8:
        TASK_SOURCE = "(!q U g) && G( !w ) && F( g && F G ( (q U (A && q )) && (q U (B && q )) ) )"  # nice dag plot
    elif MAP_NUM == 9:
        TASK_SOURCE = "G F A && G F B"
    else:
        raise ValueError("Invalid MAP_NUM")

    # -------------------------------------------------------------------------------------------
    # Parse and lower the task specification to a value tree DAG.
    logger.info("Generating the value tree DAG from logic...")
    print(f"Input task logic: {TASK_SOURCE}")

    value_tree_dag, dag_root = to_dag(TASK_SOURCE, ir_filename="rooms_discrete_ir", dag_filename="rooms_discrete_dag")
    dag_nodes = value_tree_dag.nodes

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

    map_str = [MAP0, MAP1, MAP2, MAP3, MAP4, MAP5, MAP6, MAP7, MAP8, MAP9][MAP_NUM]
    if MAP_NUM == 3:
        d_raw = parse_rooms(MAP3)[1]
        d_raw_drift = parse_rooms(MAP3_DRIFT)[1]
        d_raw = d_raw_drift | d_raw
    elif MAP_NUM == 6 or MAP_NUM == 7:
        _, d_raw = parse_rooms(map_str)

        # d_raw["q"] = d_raw["."] | d_raw["a"] | d_raw["b"]
        d_raw["q"] = d_raw["."]
        d_raw["A"] = d_raw["a"] | d_raw["A"]
        d_raw["B"] = d_raw["b"] | d_raw["B"]
        # del d_raw["a"], d_raw["b"], d_raw["."]
    else:
        _, d_raw = parse_rooms(map_str)

    key_tmp = list(d_raw.keys())[0]
    empty_map = np.zeros_like(d_raw[key_tmp])
    for ii, (k, v) in enumerate(d_raw.items()):
        empty_map = np.where(v, ii, empty_map)

    tick_locs = np.arange(len(d_raw)) + 0.5

    fig, ax = plt.subplots()
    # cmap = plt.get_cmap("seaborn", len(d_raw))
    cmap = plt.get_cmap("tab20", len(d_raw))
    colors = cmap.colors

    # Get the index of " " in the keys to set it to blank.
    if " " in d_raw:
        space_idx = list(d_raw.keys()).index(" ")
        colors[space_idx] = np.array([1.0, 1.0, 1.0, 0.0])

    if "A" in d_raw:
        space_idx = list(d_raw.keys()).index("A")
        colors[space_idx] = np.array([77 / 255, 114 / 255, 176 / 255, 1.0])

    if "B" in d_raw:
        space_idx = list(d_raw.keys()).index("B")
        colors[space_idx] = np.array([85 / 255, 168 / 255, 104 / 255, 1.0])

    if "g" in d_raw:
        space_idx = list(d_raw.keys()).index("g")
        colors[space_idx] = np.array([221 / 255, 132 / 255, 83 / 255, 1.0])

    if "C" in d_raw:
        space_idx = list(d_raw.keys()).index("C")
        colors[space_idx] = np.array([221 / 255, 132 / 255, 83 / 255, 1.0])

    if "K" in d_raw:
        space_idx = list(d_raw.keys()).index("K")
        colors[space_idx] = np.array([221 / 255, 132 / 255, 83 / 255, 1.0])

    if "D" in d_raw:
        space_idx = list(d_raw.keys()).index("D")
        colors[space_idx] = np.array([147 / 255, 120 / 255, 96 / 255, 1.0])

    if "#" in d_raw:
        space_idx = list(d_raw.keys()).index("#")
        # colors[space_idx] = np.array([220/255, 100/255, 120/255, 0.7])
        colors[space_idx] = np.array([140 / 255, 114 / 255, 179 / 255, 0.3])

    if "1" in d_raw:
        space_idx = list(d_raw.keys()).index("1")
        # muted red
        colors[space_idx] = np.array([0.8, 0.4, 0.4, 1.0])

    if "2" in d_raw:
        space_idx = list(d_raw.keys()).index("2")
        colors[space_idx] = np.array([147 / 255, 120 / 255, 96 / 255, 0.7])

    if "^" in d_raw:
        space_idx = list(d_raw.keys()).index("^")
        colors[space_idx] = np.array([0.8, 0.4, 0.4, 1.0])

    cmap = ListedColormap(colors)

    im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw), alpha=0.8, origin="lower")
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))
    ax.set_title("Map visualization")
    # Set ticks with blank labels
    ax.set_xticks(np.arange(w + 1) - 0.5, [""] * (w + 1))
    ax.set_yticks(np.arange(h + 1) - 0.5, [""] * (h + 1))
    fig.savefig("rooms_discrete.pdf")
    fig.savefig("rooms_discrete.png")
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
    sol_pkls_dir = pathlib.Path("sol_pkls")
    sol_pkls_dir.mkdir(exist_ok=True)
    pkl_path = sol_pkls_dir / "rooms_discrete_{}_gamma{}_sol.pkl".format(MAP_NUM, gamma)
    # ipdb.set_trace()

    if resolve or not pkl_path.exists():
        dict_vars, dict_actions, dict_GU_vars, dict_GU_actions = solve_discrete(
            dyn, dag_nodes, dict_predicates, gamma=gamma
        )

        # Save the solution.
        extras = {
            "task_source": TASK_SOURCE,
            "dict_predicates": dict_predicates,
            "gamma": gamma,
            "d_raw": d_raw,
            "map_num": MAP_NUM,
        }
        save_discrete_sol(
            pkl_path,
            dyn,
            dag_nodes,
            dag_root,
            dict_vars,
            dict_actions,
            dict_GU_vars,
            dict_GU_actions,
            extras=extras,
        )

    dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras = load_discrete_sol(
        pkl_path
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
    im = ax.imshow(dict_vars[dag_root].reshape(dyn.shape), vmin=-1, vmax=1, cmap="viridis")
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
        # start_state = dyn.encode_state((2, 5))
        start_state = dyn.encode_state((2, 7))
    elif MAP_NUM == 6 or MAP_NUM == 7:
        start_state = dyn.encode_state((7, 2))
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
    rollouter = MinTimeRollout(dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
    Tp1_states, T_actions, T_curnode_idxs = rollouter.rollout(start_state, max_steps=30)

    logger.info("dyn.shape: {}".format(dyn.shape))

    # Visualize the rollout by animating the path and saving as mp4.
    n_frames = len(Tp1_states)
    fig, ax = plt.subplots()

    # Visualize the map again.
    im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw), alpha=0.5, origin="lower")
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

        if kk < len(T_curnode_idxs):
            kk_text.set_text(f"Step {kk: 3} | Node {T_curnode_idxs[kk]}")
        else:
            kk_text.set_text(f"Step {kk: 3}")

        return [state_dot, kk_text]

    anim = FuncAnimation(fig, update_fn, n_frames, init_fn, blit=True)
    anim.save(results_dir / f"map{MAP_NUM}.mp4", fps=5, dpi=200)

    logger.info("Done!")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
