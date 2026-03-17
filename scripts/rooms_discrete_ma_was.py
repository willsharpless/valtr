import pathlib

import cyclopts
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.gridworld_ma import GridWorldMA, ma_collision_predicate, rew_to_ma, ma_distance_predicate
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch

from valtr.gridworld_utils import GridWorldDriftFn, parse_rooms
from valtr.mintime_rollout import MinTimeRollout
from valtr.solve_discrete import load_discrete_sol, save_discrete_sol, solve_discrete
from valtr.valtr import to_dag

plt.style.use("seaborn-v0_8-darkgrid")

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

# MAP = """
# #########################
# #     ####              #
# #  AA ####   ####       #
# #  AA ####   ####   BB  #
# #     ####   ####   BB  #
# #            ####       #
# #  ######   ######   ####
# #  ######   ######   ####
# #   ##             #    #
# #  ###  ###    #######  #
# #  ###  ### FF #######  #
# #           FF          #
# # #### ##               #
# # #### ##        #####  #
# #      ##    #   #####  #
# #  EE  #######   ###### #
# #  EE  #######   ###### #
# #                ###### #
# #  ####   CC            #
# #  ####   CC   ######   #
# #   ##         ######   #
# #  #######   #######    #
# #  #######   #######    #
# #                       #
# #########################
# """

# MAP = """
# ################
# #     B####    #
# # ###          #
# # #    ## FF   #
# # #       FF   #
# #   EE       # #
# #   EE ####### #
# #   s  ####### #
# #    s         #
# # ###     CC   #
# # ###     CC   #
# #  #  A        #
# # #######    ###
# # #######   ####
# #              #
# ################
# """

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

# TASK_SOURCE = "G F r1 && G F r2 && G F r3 && (!d U k) && G(!w) && G(!collide)"
# TASK_SOURCE = "F r1 && F r2 && G F r3 && G F r4 && F G r5 && G(!w) && G(!collide)"
# TASK_SOURCE = "F r1 && F r2 && G F r4 && G F r3 && G F r5 && G(!w) && G(!collide)"
TASK_SOURCE = "(!site U gear) && G F saw && G F wood && (!d U k) && G(!w) && G(!collide) && G(!distant)"
# TASK_SOURCE = "G(!w) && G(!collide)"


def draw_room_map(fig, ax, empty_map, cmap, d_raw, w, h, alpha: float = 0.5):
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(empty_map, cmap=cmap, vmin=0, vmax=len(d_raw), alpha=alpha, origin="lower")
    # cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(d_raw)) + 0.5)
    # cbar.ax.set_yticklabels(list(d_raw.keys()))
    ax.set_xticks(np.arange(w + 1) - 0.5)
    ax.set_yticks(np.arange(h + 1) - 0.5)
    # ax.grid(color="white", linestyle="-", linewidth=1., alpha=0.3)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.margins(0)
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    return im


@app.default()
def main(view_pdf: bool = False, gamma: float | None = None, resolve: bool = False):
    dyn, d_raw = parse_rooms(MAP)
    empty_mask = np.zeros_like(d_raw["#"], dtype=bool)
    get_mask = lambda key: d_raw.get(key, empty_mask)
    d = {
        "r1": np.where(get_mask("A"), 1, -1),
        "r2": np.where(get_mask("B"), 1, -1),
        "r3": np.where(get_mask("C"), 1, -1),
        "saw": np.where(get_mask("E"), 1, -1),
        "wood": np.where(get_mask("F"), 1, -1),
        "gear": np.where(get_mask("g"), 1, -1),
        "site": np.where(get_mask("."), 1, -1),
        "k": np.where(get_mask("K"), 1, -1),
        "d": np.where(get_mask("D"), 1, -1),
        "w": np.where(get_mask("#"), 1, -1),
        # Just for convenience.
        "<": np.where(get_mask("<"), 1, -1),
        ">": np.where(get_mask(">"), 1, -1),
        "^": np.where(get_mask("^"), 1, -1),
    }
    if any(key in d_raw for key in ("<", ">", "^", "v")):
        dyn.drift_fn = GridWorldDriftFn(d_raw, force=False)
    # ------------------------------

    h, w = dyn.shape
    empty_map = np.zeros_like(d_raw["#"])
    space_idx = list(d_raw.keys()).index(" ") if " " in d_raw else 0
    for ii, (k, v) in enumerate(d_raw.items()):
        if k == "s":
            empty_map = np.where(v, space_idx, empty_map)
        else:
            empty_map = np.where(v, ii, empty_map)
    cmap = plt.get_cmap("tab10", len(d_raw))
    base_colors = np.array(cmap.colors, copy=True)
    colors = cmap.colors

    # Get the index of " " in the keys to set it to white.
    if " " in d_raw:
        colors[space_idx] = np.array([1.0, 1.0, 1.0, 0.0])

    if "." in d_raw:
        space_idx = list(d_raw.keys()).index(".")
        # colors[space_idx] = np.array([1.0, 1.0, 1.0, 0.0])
        # colors[space_idx] = np.array([77/255, 114/255, 176/255, 0.3]) # muted royal blue
        colors[space_idx] = np.array([227/255, 197/255, 87/255, 0.5]) # muted yellow

    if "s" in d_raw:
        s_idx = list(d_raw.keys()).index("s")
        colors[s_idx] = colors[space_idx]

    if "A" in d_raw:
        space_idx = list(d_raw.keys()).index("A")
        # colors[space_idx] = np.array([77/255, 114/255, 176/255, 1.0]) # muted royal blue
        colors[space_idx] = np.array([140/255, 114/255, 179/255, 1.0]) # muted purple

    if "B" in d_raw:
        space_idx = list(d_raw.keys()).index("B")
        # colors[space_idx] = np.array([85/255, 168/255, 104/255, 1.0]) # muted green
        # colors[space_idx] = np.array([221/255, 132/255, 83/255, 1.0]) # muted orange
        colors[space_idx] = np.array([147/255, 120/255, 96/255, 1.0]) # muted brown

    if "C" in d_raw:
        space_idx = list(d_raw.keys()).index("C")
        colors[space_idx] = np.array([221/255, 132/255, 83/255, 1.0]) # muted orange

    if "E" in d_raw:
        space_idx = list(d_raw.keys()).index("E")
        colors[space_idx] = np.array([85/255, 168/255, 104/255, 1.0]) # muted green

    if "F" in d_raw:
        space_idx = list(d_raw.keys()).index("F")
        colors[space_idx] = np.array([0.8, 0.4, 0.4, 1.0]) # muted red

    if "K" in d_raw:
        space_idx = list(d_raw.keys()).index("K")
        colors[space_idx] = np.array([221/255, 132/255, 83/255, 1.0]) # muted orange

    if "d" in d_raw:
        space_idx = list(d_raw.keys()).index("d")
        colors[space_idx] = np.array([0.05, 0.05, 0.05, 1.0]) # dark grey

    if "g" in d_raw:
        space_idx = list(d_raw.keys()).index("g")
        # colors[space_idx] = np.array([227/255, 197/255, 87/255, 1.0]) # muted yellow
        colors[space_idx] = np.array([77/255, 114/255, 176/255, 1.]) # muted royal blue

    if "#" in d_raw:
        # import seaborn as sns

        space_idx = list(d_raw.keys()).index("#")
        # colors[space_idx] = np.array([220/255, 100/255, 120/255, 0.7]) # dark pink
        # colors[space_idx] = np.array([140/255, 114/255, 179/255, 0.3]) # muted purple
        colors[space_idx] = np.array([0.33714769, 0.41920711, 0.54334937, 1.0]) # charcoal gray
        # palette = sns.dark_palette("#79C")
        # sns_color = palette[-3] # -2 kinda good but need to change blue if so
        # colors[space_idx, :3] = sns_color
        # colors[space_idx, 3] = 0.8

    if "1" in d_raw:
        space_idx = list(d_raw.keys()).index("1")
        colors[space_idx] = np.array([0.8, 0.4, 0.4, 1.0]) # muted red

    if "2" in d_raw:
        space_idx = list(d_raw.keys()).index("2")
        colors[space_idx] = np.array([147/255, 120/255, 96/255, 0.7]) # muted brown


    cmap = ListedColormap(colors)

    # ------------------------------
    # dyn_ma = GridWorldMA(dyn, n_agents=1)
    dyn_ma = GridWorldMA(dyn, n_agents=2)
    # dyn_ma = GridWorldMA(dyn, n_agents=3)

    value_tree_dag, dag_root = to_dag(
        TASK_SOURCE, ir_filename="rooms_discrete_ma_ir", dag_filename="rooms_discrete_ma_dag"
    )
    dag_nodes = value_tree_dag.nodes

    dict_predicates_unflat = d
    dict_predicates = {k: v.flatten() for k, v in dict_predicates_unflat.items()}

    # Convert from single-agent to multi-agent predicates.
    collide_dist = 1.0  # Diagonal is safe, but not adjacent.
    dict_predicates = {
        "r1": rew_to_ma(dict_predicates["r1"], dyn_ma.n_agents, "max"),
        "r2": rew_to_ma(dict_predicates["r2"], dyn_ma.n_agents, "max"),
        "r3": rew_to_ma(dict_predicates["r3"], dyn_ma.n_agents, "max"),
        "saw": rew_to_ma(dict_predicates["saw"], dyn_ma.n_agents, "min"),
        "wood": rew_to_ma(dict_predicates["wood"], dyn_ma.n_agents, "min"),
        "gear": rew_to_ma(dict_predicates["gear"], dyn_ma.n_agents, "max"),
        "site": rew_to_ma(dict_predicates["site"], dyn_ma.n_agents, "max"),
        "k": rew_to_ma(dict_predicates["k"], dyn_ma.n_agents, "max"),
        "d": rew_to_ma(dict_predicates["d"], dyn_ma.n_agents, "max"),
        "w": rew_to_ma(dict_predicates["w"], dyn_ma.n_agents, "max"),
        "collide": ma_collision_predicate(dyn_ma, collide_dist),
        "distant": ma_distance_predicate(dyn_ma, 2*collide_dist),
    }

    # -------------------------------------------
    # Solve.
    pkl_path = pathlib.Path("rooms_discrete_ma_sol.pkl")

    if resolve or not pkl_path.exists():
        dict_vars, dict_actions, dict_GU_vars, dict_GU_actions = solve_discrete(
            dyn_ma, dag_nodes, dict_predicates, gamma=gamma
        )
        extras = {
            "task_source": TASK_SOURCE,
            "dict_predicates": dict_predicates,
            "gamma": gamma,
            "d_raw": d_raw,
            # "map_num": MAP_NUM,
        }
        save_discrete_sol(pkl_path, dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras=extras)

    dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras = load_discrete_sol(pkl_path)

    # ---------------------------------
    rng = np.random.default_rng(seed=12345)

    value = dict_vars[dag_root]

    # Visualize the root-node value with the other agent fixed at selected positions.
    if "s" in d_raw and np.any(d_raw["s"]):
        fixed_positions = [tuple(pos) for pos in np.argwhere(d_raw["s"])]
    else:
        fixed_positions = [(4, 4), (11, 11)]
    fig_values, axes_values = plt.subplots(1, len(fixed_positions), figsize=(6 * len(fixed_positions), 5), layout="constrained")
    if len(fixed_positions) == 1:
        axes_values = [axes_values]

    root_value_maps = []
    for fixed_pos in fixed_positions:
        value_map = np.full(dyn.shape, np.nan, dtype=float)
        for y in range(h):
            for x in range(w):
                if d_raw["#"][y, x]:
                    continue
                joint_state = int(dyn_ma.encode_from_tups([(y, x), fixed_pos]))
                value_map[y, x] = float(value[joint_state])
        root_value_maps.append(value_map)

    finite_values = np.concatenate([value_map[np.isfinite(value_map)] for value_map in root_value_maps])
    vmin = float(finite_values.min()) if finite_values.size else -1.0
    vmax = float(finite_values.max()) if finite_values.size else 1.0

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

    fig_values.colorbar(im_values, ax=axes_values, fraction=0.046, pad=0.04)
    fig_values.savefig("rooms_discrete_ma_root_values.png", dpi=200, bbox_inches="tight")
    plt.close(fig_values)

    feasible_states = np.where(value >= 0)[0]
    logger.info("Num feasible states: {}".format(len(feasible_states)))

    is_good = (dict_predicates["k"] != 1) & (value >= 0)
    feasible_states_good = np.where(is_good)[0]
    logger.info("Num feasible states (where not on key): {}".format(len(feasible_states_good)))

    # If possible, choose an initial state where none of the agents start on the key.
    if 's' in d_raw and np.any(d_raw['s']): # parse from MAP at 's'
        start_states = [tuple(pos) for pos in np.argwhere(d_raw['s'])][:2]
        start_state = dyn_ma.encode_from_tups(start_states)
    else: # hardcode
        # start_state = dyn_ma.encode_from_tups([(3, 3), (3, 5), (7, 5)])
        start_state = dyn_ma.encode_from_tups([(3, 3), (12, 12)])
        # start_state = dyn_ma.encode_from_tups([(6, 6)])
    if start_state is not None and value[start_state] >= 0:
        logger.info("Using hardcoded start state.")
    else:
        logger.info("Hardcoded start state not feasible: {}".format(value[start_state]))
        if len(feasible_states_good) > 0:
            start_state = rng.choice(feasible_states_good)
        else:
            start_state = rng.choice(feasible_states)

    # ---------------------------------
    rollouter = MinTimeRollout(dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
    Tp1_states, T_actions, T_curnode_idxs = rollouter.rollout(start_state, max_steps=100)

    # Visualize the rollout by animating the path and saving as mp4.
    n_frames = len(Tp1_states)
    fig_anim, ax_anim = plt.subplots(frameon=False)

    # Visualize the map again.
    draw_room_map(fig_anim, ax_anim, empty_map, cmap, d_raw, w, h, alpha=0.5)

    # --- Multi-agent: one dot per agent ---
    n_agents = dyn_ma.n_agents
    agent_colors = ["C0", "C1", "C2"]

    agent_dots = []
    for i in range(n_agents):
        (dot,) = ax_anim.plot([], [], marker="o", color=agent_colors[i], ms=5, linestyle="None")
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

    anim = FuncAnimation(fig_anim, update_fn, n_frames, init_fn, blit=True)
    anim.save("rooms_discrete_rollout_multiagent.mp4", fps=5, dpi=200)
    plt.close(fig_anim)

    # Save a still image with the first part of the path and one arrow per transition.
    fig_still, ax_still = plt.subplots(frameon=False, figsize=(4, 4))
    draw_room_map(fig_still, ax_still, empty_map, cmap, d_raw, w, h, alpha=0.5)

    steps_to_plot = 50
    n_plot_steps = min(steps_to_plot, len(Tp1_states))
    Tp1_states_plot = Tp1_states[:n_plot_steps]
    T_curnode_idxs_plot = T_curnode_idxs[: max(n_plot_steps - 1, 0)]

    decoded_joint_states = np.array([dyn_ma.decode_joint_state(int(joint_state)) for joint_state in Tp1_states_plot])
    node_cmap = plt.get_cmap("tab20", max(len(dag_nodes), 1))
    unique_node_ids = np.unique(T_curnode_idxs_plot)
    offset_radius = 0.14
    if n_agents == 1:
        agent_plot_offsets = np.zeros((1, 2))
    elif n_agents == 2:
        agent_plot_offsets = offset_radius * np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
        ]) / np.sqrt(2.0)
    else:
        offset_angles = np.linspace(np.pi / 4.0, np.pi / 4.0 + 2.0 * np.pi, n_agents, endpoint=False)
        agent_plot_offsets = offset_radius * np.column_stack((np.cos(offset_angles), np.sin(offset_angles)))

    for i in range(n_agents):
        agent_state_traj = decoded_joint_states[:, i]
        coords = np.array([dyn_ma.base.decode_state(int(state)) for state in agent_state_traj])
        offset_x, offset_y = agent_plot_offsets[i]
        ys = coords[:, 0] + offset_y
        xs = coords[:, 1] + offset_x
        color = agent_colors[i]

        ax_still.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, zorder=4)
        for step_idx in range(len(xs) - 1):
            start = np.array([xs[step_idx], ys[step_idx]], dtype=float)
            end = np.array([xs[step_idx + 1], ys[step_idx + 1]], dtype=float)
            delta = end - start
            if np.allclose(delta, 0.0):
                continue
            node_color = node_cmap(int(T_curnode_idxs_plot[step_idx]))
            edge_rgba = to_rgba(color, alpha=0.3)
            face_rgba = to_rgba(node_color, alpha=1.0)
            direction = delta / np.linalg.norm(delta)
            shrink = 0.08 * direction
            arrow = FancyArrowPatch(
                posA=tuple(start + shrink),
                posB=tuple(end - shrink),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=0.5,
                # edgecolor=edge_rgba,
                facecolor=face_rgba,
                zorder=5,
            )
            ax_still.add_patch(arrow)
        ax_still.plot(xs[0], ys[0], marker="o", color=color, ms=6, zorder=6)
        ax_still.plot(xs[-1], ys[-1], marker="s", color=color, ms=5, zorder=6)

    # if len(unique_node_ids) > 0:
    #     node_norm = BoundaryNorm(np.arange(len(dag_nodes) + 1) - 0.5, node_cmap.N)
    #     sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    #     sm.set_array([])
    #     cbar = fig_still.colorbar(sm, ax=ax_still, ticks=unique_node_ids, fraction=0.046, pad=0.04)
    #     cbar.ax.set_ylabel("Value tree node")

    # ax_still.set_title(f"Multi-agent rollout paths (first {n_plot_steps} states)")
    fig_still.savefig("rooms_discrete_rollout_multiagent_paths.png", dpi=200, pad_inches=0)
    plt.close(fig_still)

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()