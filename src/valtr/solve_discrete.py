import functools as ft
import jax
import pathlib
import pickle
from typing import NamedTuple

import ipdb
import numpy as np
import tqdm
from dvi.dynamics.discrete import DiscreteDyn
from dvi.gen_solver import (FixedPointGUSolver, avoid_update_rule_with_actions, make_solve_fn_with_actions,
                            reach_avoid_update_rule_with_actions)
from loguru import logger

from valtr.reachability import (DAGGU, DAGAvoid, DagBuilder, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode, DAGReachAvoid,
                                DAGVar)


def solve_discrete(
    dyn: DiscreteDyn, dag_nodes: list[DAGNode], dict_predicates: dict[str, np.ndarray], gamma: float | None = None
):
    dict_vars = {}
    dict_actions = {}
    dict_GU_vars = {}
    dict_GU_actions = {}
    dict_locals = dict_predicates
    pbar = tqdm.tqdm(dag_nodes)
    for dag_id, node in enumerate(pbar):
        pbar.set_description(f"Solving node {type(node)}")
        match node:
            case DAGVar(name=name):
                assert name in dict_locals, "Unknown variable name {}".format(name)
                dict_vars[dag_id] = dict_locals[name]
                assert dict_vars[dag_id].shape == (dyn.n_states,)

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
                update_rule = reach_avoid_update_rule_with_actions(gamma)
                solve_fn = make_solve_fn_with_actions(dyn, update_rule, n_updates=dyn.n_states)
                dict_vars[dag_id], dict_actions[dag_id] = solve_fn(s_v0, **kwargs)

            case DAGAvoid(avoid=avoid):
                # Note: the avoid is a stay since we are maximizing the value.
                arg_avoid = dict_vars[avoid]

                s_v0 = arg_avoid
                kwargs = dict(s_q=arg_avoid)

                # update_rule = avoid_update_rule
                # solve_fn = make_solve_fn(dyn, update_rule, n_updates=dyn.n_states)
                update_rule = avoid_update_rule_with_actions(gamma)
                solve_fn = make_solve_fn_with_actions(dyn, update_rule, n_updates=dyn.n_states)
                dict_vars[dag_id], dict_actions[dag_id] = solve_fn(s_v0, **kwargs)

            case DAGGU(args=args):
                U_args = [[dict_vars[q], dict_vars[r]] for q, r in args]
                out = FixedPointGUSolver().solve(dyn, U_args, n_iters=3, gamma=gamma)
                dict_vars[dag_id], dict_actions[dag_id], dict_GU_vars[dag_id], dict_GU_actions[dag_id] = out

        # # Visualize.
        # import matplotlib.pyplot as plt
        #
        # tmp = dict_vars[dag_id]
        # vmin = tmp[tmp >= 0].min()
        # if vmin == 1:
        #     vmin = -1
        #
        # h, w = dyn.shape
        # fig, ax = plt.subplots()
        # im = ax.imshow(dict_vars[dag_id].reshape(dyn.shape), vmin=vmin, vmax=1)
        #
        # ax.set_xticks(np.arange(w + 1) - 0.5)
        # ax.set_yticks(np.arange(h + 1) - 0.5)
        # cbar = fig.colorbar(im, ax=ax)
        # ax.set_title("{} ({}), gamma={}".format(dag_id, node, gamma))
        # plt.show()

    return dict_vars, dict_actions, dict_GU_vars, dict_GU_actions


class DiscreteSol(NamedTuple):
    dyn_ma: DiscreteDyn
    dag_nodes: list[DAGNode]
    dag_root: DAGId
    dict_vars: dict[DAGId, np.ndarray]
    dict_actions: dict[DAGId, np.ndarray]
    dict_GU_vars: dict[DAGId, np.ndarray]
    dict_GU_actions: dict[DAGId, np.ndarray]


def save_discrete_sol(
    pkl_path: pathlib.Path,
    dyn_ma: DiscreteDyn,
    dag_nodes: list[DAGNode],
    dag_root: DAGId,
    dict_vars: dict[DAGId, np.ndarray],
    dict_actions: dict[DAGId, np.ndarray],
    dict_GU_vars: dict[DAGId, np.ndarray],
    dict_GU_actions: dict[DAGId, np.ndarray],
):
    out_dict = {
        "dyn_ma": dyn_ma,
        "dag_nodes": dag_nodes,
        "dag_root": dag_root,
        "dict_vars": jax.device_get(dict_vars),
        "dict_actions": jax.device_get(dict_actions),
        "dict_GU_vars": jax.device_get(dict_GU_vars),
        "dict_GU_actions": jax.device_get(dict_GU_actions),
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(out_dict, f)

    logger.success("Saved discrete solution to {}".format(pkl_path))


def load_discrete_sol(pkl_path: pathlib.Path):
    with open(pkl_path, "rb") as f:
        in_dict = pickle.load(f)

    dyn_ma = in_dict["dyn_ma"]
    dag_nodes = in_dict["dag_nodes"]
    dag_root = in_dict["dag_root"]
    dict_vars = in_dict["dict_vars"]
    dict_actions = in_dict["dict_actions"]
    dict_GU_vars = in_dict["dict_GU_vars"]
    dict_GU_actions = in_dict["dict_GU_actions"]

    logger.success("Loaded discrete solution from {}".format(pkl_path))

    return DiscreteSol(dyn_ma, dag_nodes, dag_root, dict_vars, dict_actions, dict_GU_vars, dict_GU_actions)
