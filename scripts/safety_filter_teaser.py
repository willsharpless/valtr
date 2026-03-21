import pathlib

import cyclopts
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.gridworld import GridWorld
from dvi.dynamics.gridworld_timed import GridWorldTimed
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from valtr.gridworld_utils import GridWorldDriftFn, GridWorldDriftFn2, parse_rooms
from valtr.mintime_policy import MinTimePolicy
from valtr.mintime_rollout import MinTimeRollout
from valtr.safety_filter import SafetyFilter
from valtr.solve_discrete import load_discrete_sol, save_discrete_sol, solve_discrete
from valtr.valtr import to_dag

plt.style.use("seaborn-v0_8-darkgrid")

app = cyclopts.App()

# MAP = """
# ###########
# #      #  #
# ####   #W.#
# #  #  kd S#
# #W d   ####
# #.S#      #
# ###########
# """

MAP = """
#################################
#################################
#################################
###        #   ___   ###      ###
###        #   _s_   ###      ###
###        #   ___   ###      ###
############         ###-b----###
############         ###-    -###
############         ###---- -###
###      ###         ###   - v###
###      ###       k ddd   ^ -###
###      ###         ###   -a-###
###.B.   ###         ############
###. v   ddd         ############
###. .   ###         ############
###^ ....###                  ###
###.    A###                  ###
###......###                  ###
#################################
#################################
#################################
"""


def get_rooms():
    s = MAP

    dyn, d_raw = parse_rooms(s, flipy=True)
    d_raw["wall"] = d_raw.pop("#")
    d_raw["safe"] = ~d_raw["wall"]

    d_raw["leftsaw"] = d_raw["A"]
    d_raw["saw"] = d_raw["a"] | d_raw["A"]
    d_raw["wood"] = d_raw["b"] | d_raw["B"]

    d = {
        "key": np.where(d_raw["k"], 1, -1),
        "door": np.where(d_raw["d"], 1, -1),
        "wood": np.where(d_raw["wood"], 1, -1),
        "saw": np.where(d_raw["saw"], 1, -1),
        "leftsaw": np.where(d_raw["leftsaw"], 1, -1),
        "safe": np.where(d_raw["safe"], 1, -1),
        "site": np.where(d_raw["."] | d_raw["-"] | d_raw["saw"] | d_raw["wood"] | d_raw["^"] | d_raw["v"], 1, -1),
    }

    return dyn, d


def _setup_ax(ax, h, w):
    ax.set_aspect("equal")
    ax.grid(which="major", visible=False)
    ax.grid(which="minor", visible=True)
    ax.set_xticks(np.arange(h + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(w + 1) - 0.5, minor=True)
    ax.set(xlabel="x", ylabel="y")


def _make_overlay_text(ax, x, y, ha, va):
    return ax.text(
        x,
        y,
        "",
        transform=ax.transAxes,
        verticalalignment=va,
        horizontalalignment=ha,
        color="white",
        fontsize=8,
        fontfamily=["DejaVu Sans", "Noto Sans"],
        bbox=dict(facecolor="black", alpha=0.5, pad=2),
    )


RESULTS_DIR = pathlib.Path("plots_discrete_teaser")
RESULTS_DIR.mkdir(exist_ok=True)


def _solve_and_cache(dyn, dag_nodes, dag_root, dict_predicates, pkl_path, task_source, d_raw, gamma, resolve):
    RESULTS_DIR.mkdir(exist_ok=True)
    if resolve or not pkl_path.exists():
        dict_vars, dict_actions, dict_GU_vars, dict_GU_actions = solve_discrete(
            dyn, dag_nodes, dict_predicates, gamma=gamma
        )
        extras = {"task_source": task_source, "dict_predicates": dict_predicates, "gamma": gamma, "d_raw": d_raw}
        save_discrete_sol(
            pkl_path, dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras=extras
        )
    return load_discrete_sol(pkl_path)


@app.default()
def main(
    gamma: float | None = None,
    resolve: bool = False,
    resolve_nom: bool = False,
    use_filter: bool = False,
    nominal: bool = False,
    safetyonly: bool = False,
):
    TMAX = 30
    TASK_SOURCE = "G( safe ) && (!door U (key && TleKey)) && ( TleSite U ( G( site ) ) ) && G( F wood && F saw )"
    # TASK_SOURCE = "G( safe ) && (!door U (key && TleKey))"

    if safetyonly:
        TASK_SOURCE = "G( safe && !door )"

    NOM_TASK_SOURCE = "F leftsaw && G( F wood && F saw ) && F( G( site ) )"
    # NOM_TASK_SOURCE = "F leftsaw"

    tag = "safetyonly" if safetyonly else "full"

    value_tree_dag, dag_root = to_dag(TASK_SOURCE, dag_filename=str(RESULTS_DIR / f"{tag}_dag"))
    dag_nodes = value_tree_dag.nodes

    _, d_raw = parse_rooms(MAP, flipy=True)
    d_raw["w"] = d_raw["#"]

    drift_fn = GridWorldDriftFn2(d_raw, force=True)

    dyn_: GridWorld
    dyn_, dict_predicates_unflat = get_rooms()
    dyn_.drift_fn = drift_fn
    dyn_.allow_diagonal = True

    h, w = dyn_.shape

    dyn = GridWorldTimed(dyn_.shape, t_max=TMAX, drift_fn=drift_fn, allow_diagonal=True, freeze_at_t_max=False)

    dict_predicates = {k: v.flatten() for k, v in dict_predicates_unflat.items()}

    # ---------------------------
    # Rename.
    d_raw["wall"] = d_raw.pop("#")

    key_tmp = list(d_raw.keys())[0]
    empty_map = np.zeros_like(d_raw[key_tmp])
    for ii, (k, v) in enumerate(d_raw.items()):
        empty_map = np.where(v, ii, empty_map)

    tick_locs = np.arange(len(d_raw)) + 0.5
    # -----------------------------------
    # Visualize. Draw the map.
    cmap = plt.get_cmap("tab20", len(d_raw))
    colors = cmap.colors
    if " " in d_raw:
        space_idx = list(d_raw.keys()).index(" ")
        colors[space_idx] = np.array([1.0, 1.0, 1.0, 0.0])
    if "w" in d_raw:
        space_idx = list(d_raw.keys()).index("w")
        colors[space_idx] = np.array([140 / 255, 114 / 255, 179 / 255, 0.3])

    cmap = ListedColormap(colors)

    fig, ax = plt.subplots()
    im = ax.imshow(empty_map.T, cmap=cmap, vmin=0, vmax=len(d_raw), origin="lower")

    state = dyn.encode_state((5, 5, 5))
    x0, y0, t0 = dyn.decode_state(state)
    action = dyn.str_to_action("R")
    state_new = dyn.step(state, action)
    x1, y1, t1 = dyn.decode_state(state_new)

    ax.plot([x0], [y0], marker="o", color="C0")
    ax.plot([x1], [y1], marker="s", color="C1")

    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))
    _setup_ax(ax, h, w)
    fig.savefig(RESULTS_DIR / "map.pdf")
    # ---------------------------

    dict_predicates_timed = {k: ei.repeat(v, "x -> (x T)", T=TMAX + 1) for k, v in dict_predicates.items()}
    dict_predicates_timed["TleKey"] = dyn.tle_predicate(8)
    dict_predicates_timed["TleSite"] = dyn.tle_predicate(22)
    # dict_predicates_timed["TleKey"] = dyn.tle_predicate(15)

    # -------------------------------------------
    # Solve.
    pkl_path = RESULTS_DIR / f"{tag}_sol.pkl"

    dyn: GridWorldTimed
    dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions, extras = _solve_and_cache(
        dyn, dag_nodes, dag_root, dict_predicates_timed, pkl_path, TASK_SOURCE, d_raw, gamma, resolve
    )

    # ----------------------------
    # Solve for the nominal policy.
    def solve_ag1_policy(gamma: float, resolve_nom: bool):
        # Untimed.
        value_tree_dag_, dag_root_ = to_dag(
            NOM_TASK_SOURCE,
            dag_filename=RESULTS_DIR / f"ag_dag",
        )
        pkl_path_ = RESULTS_DIR / "ag_sol.pkl"
        dag_nodes_ = value_tree_dag_.nodes
        _, _, _, sol_dict_vars, sol_dict_actions, sol_dict_GU_vars, sol_dict_GU_actions, _ = _solve_and_cache(
            dyn_, dag_nodes_, dag_root_, dict_predicates, pkl_path_, NOM_TASK_SOURCE, d_raw, gamma, resolve_nom
        )
        return MinTimePolicy(
            dyn_,
            dag_nodes_,
            dag_root_,
            sol_dict_vars,
            sol_dict_actions,
            sol_dict_GU_vars,
            sol_dict_GU_actions,
        )

    pol_ag1 = solve_ag1_policy(gamma=gamma, resolve_nom=resolve_nom)

    # ----------------------------
    # Visualize the value function at the root.
    T_dict_vars = {k: v.reshape(dyn.shape) for k, v in dict_vars.items()}
    T_dict_actions = {k: v.reshape(dyn.shape) for k, v in dict_actions.items()}
    T_val_root = T_dict_vars[dag_root]

    times = [0, 1, 5, 6, TMAX - 2, TMAX - 1, TMAX]
    nrow = len(times)
    figsize = np.array([6, 2 * nrow])
    fig, axes = plt.subplots(nrow, layout="constrained", figsize=figsize)
    for ii, ax in enumerate(axes):
        val_root = T_val_root[..., times[ii]]
        im = ax.imshow(val_root.T, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set_title("Root Value function, t={}".format(times[ii]))
        _setup_ax(ax, h, w)

    fig.savefig(RESULTS_DIR / f"{tag}_val_root.pdf")

    n_valid = np.sum(T_val_root[..., 0] > 0)
    logger.info(f"Number of valid states at t=0: {n_valid}")
    if n_valid == 0:
        logger.warning("Value function is invalid everywhere.")
        ipdb.set_trace()

    # ----------------------------
    pos = None
    if "s" in d_raw and np.any(d_raw["s"]):
        pos = np.argmax(d_raw["s"])
    assert pos is not None

    state = dyn.encode_time(pos, 0)

    # Initial state should be valid.
    if T_val_root.flatten()[state] <= 0:
        logger.warning("Initial state is invalid according to value function.")
        ipdb.set_trace()

    # a_nom = dyn.str_to_action(".")

    def preference_fn(state_, a_nom_):
        deltas = np.array(dyn.make_action_deltas(which=np))
        delta_nom = deltas[a_nom_]

        costs = np.zeros(dyn.n_actions, dtype=np.float32)
        for aa in range(dyn.n_actions):
            delta_aa = deltas[aa]

            # If the action is the same as the nominal, then zero cost.
            # If there is one delta that is the same, then cost 1.
            # If both deltas are different, then cost 2.
            n_same = np.sum((delta_nom == delta_aa) & (delta_nom != 0))
            if n_same == 2:
                cost = 0.0
            elif n_same == 1:
                cost = 1.0
            else:
                cost = 2.0
            costs[aa] = cost

        return costs

    if use_filter or nominal:
        Tp1_states = [state]
        T_a_nom = []
        T_a_filt = []
        T_hasfiltered = []

        safety_filter = SafetyFilter(dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)

        # Rollout safety filter.
        for kk in range(2 * TMAX):
            logger.debug(f"> kk={kk}")

            x, y, tt = dyn.decode_state(state)
            state_untimed = dyn_.encode_state((x, y))

            a_nom, isdone = pol_ag1.get_action(state_untimed, which=np, kk=kk)

            if isdone:
                a_nom = dyn.str_to_action(".")

            if nominal:
                a_safe = a_nom
                hasfiltered = False
            else:
                a_safe = safety_filter.filter_action(state, a_nom, preference_fn=preference_fn)
                hasfiltered = a_safe != a_nom

            state = dyn.step(state, a_safe)

            T_a_nom.append(a_nom)
            T_a_filt.append(a_safe)
            T_hasfiltered.append(hasfiltered)
            Tp1_states.append(state)

        T_hasfiltered = np.array(T_hasfiltered)
    else:
        rollouter = MinTimeRollout(dyn, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
        Tp1_states, T_actions, T_curnode_idxs = rollouter.rollout(state, max_steps=2 * TMAX)
        T_a_nom = T_actions
        T_a_filt = T_actions
        T_hasfiltered = np.zeros_like(T_a_nom, dtype=bool)

    # -----------------------------------
    # Animate the rollout.
    n_frames = len(Tp1_states) - 1
    fig, ax = plt.subplots()
    im = ax.imshow(empty_map.T, cmap=cmap, vmin=0, vmax=len(d_raw), origin="lower")
    cbar = fig.colorbar(im, ax=ax, ticks=tick_locs)
    cbar.ax.set_yticklabels(list(d_raw.keys()))
    _setup_ax(ax, h, w)

    (agent_dot,) = ax.plot([], [], marker="o", color="C0", ms=5, linestyle="None")
    kk_text = _make_overlay_text(ax, 0.02, 0.98, ha="left", va="top")
    debug_text = _make_overlay_text(ax, 0.98, 0.02, ha="right", va="bottom")

    def init_fn():
        return [agent_dot, kk_text, debug_text]

    def update_fn(kk: int) -> list[plt.Artist]:
        state = Tp1_states[kk]
        x, y, t = dyn.decode_state(state)
        agent_dot.set_data([x], [y])
        kk_text.set_text(f"Step {kk: 3}")
        a_nom_str = dyn.action_to_str(T_a_nom[kk])
        a_safe_str = dyn.action_to_str(T_a_filt[kk])
        has_filtered = int(T_hasfiltered[kk])
        debug_text.set_text(
            "Timer : {}\nNom : {}\nSafe: {}\nfiltered: {}".format(t, a_nom_str, a_safe_str, has_filtered)
        )
        return [agent_dot, kk_text, debug_text]

    anim = FuncAnimation(fig, update_fn, n_frames, init_fn, blit=True)

    if use_filter:
        name = f"{tag}_filt_rollout.mp4"
    elif nominal:
        name = f"{tag}_nom_rollout.mp4"
    else:
        name = f"{tag}_opt_rollout.mp4"

    anim.save(RESULTS_DIR / name, fps=10, dpi=200)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
