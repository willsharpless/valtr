import numpy as np
import tqdm
from dvi.dynamics.discrete import DiscreteDyn
from dvi.gen_solver import (FixedPointGUSolver, avoid_update_rule_with_actions, make_solve_fn_with_actions,
                            reach_avoid_update_rule_with_actions)

from valtr.reachability import DAGGU, DAGAvoid, DagBuilder, DAGMaxN, DAGMinN, DAGNegate, DAGReachAvoid, DAGVar


def solve_discrete(dyn: DiscreteDyn, value_tree_dag: DagBuilder, dict_predicates: dict[str, np.ndarray]):
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

    return dict_vars, dict_actions, dict_GU_vars, dict_GU_actions
