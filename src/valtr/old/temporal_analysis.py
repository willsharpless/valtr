import ipdb
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from .reachability import (DAGAvoid, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGReach, DAGReachAvoid, DAGVar, has_temporal_children)


def evaluate_ltl_finite_dag(dag_nodes: list[DAGId], dag_root, T_pred: dict[str, np.ndarray], which=jnp):
    return evaluate_ltl_finite_dag_iterative(dag_nodes, dag_root, T_pred, which=which)


def evaluate_ltl_finite_dag_iterative(dag_nodes: list[DAGId], dag_root: DAGId, T_pred: dict[str, np.ndarray], which=np):
    sample = next(iter(T_pred.values()))
    (T,) = sample.shape
    N = len(dag_nodes)

    curr = [None] * N
    next_vals = [None] * N

    for t in tqdm.trange(T - 1, -1, -1, leave=False):
        is_terminal = t == T - 1

        for node_id, node in enumerate(dag_nodes):
            match node:
                case DAGConst(value=value):
                    val = which.full((), value)

                case DAGVar(name=name):
                    val = T_pred[name][t]

                case DAGNegate(arg=arg_id):
                    val = -curr[arg_id]

                case DAGMaxN(args=args_ids):
                    val = curr[args_ids[0]]
                    for arg_id in args_ids[1:]:
                        val = which.maximum(val, curr[arg_id])

                case DAGMinN(args=args_ids):
                    val = curr[args_ids[0]]
                    for arg_id in args_ids[1:]:
                        val = which.minimum(val, curr[arg_id])

                case DAGReach(reach=reach_id):
                    reach_val = curr[reach_id]
                    val = reach_val if is_terminal else which.maximum(reach_val, next_vals[node_id])

                case DAGAvoid(avoid=stay_id):
                    stay_val = curr[stay_id]
                    val = stay_val if is_terminal else which.minimum(stay_val, next_vals[node_id])

                case DAGReachAvoid(reach=reach_id, avoid=stay_id):
                    reach_val = curr[reach_id]
                    stay_val = curr[stay_id]
                    val = (
                        reach_val
                        if is_terminal
                        else which.maximum(reach_val, which.minimum(stay_val, next_vals[node_id]))
                    )

                case DAGGUMinN(args=args_ids):
                    val = curr[args_ids[0]]
                    for arg_id in args_ids[1:]:
                        val = which.minimum(val, curr[arg_id])

                case DAGGUSingle(reach=reach_id, avoid=stay_id):
                    reach_val = curr[reach_id]
                    stay_val = curr[stay_id]
                    val = (
                        reach_val
                        if is_terminal
                        else which.maximum(reach_val, which.minimum(stay_val, next_vals[node_id]))
                    )

                case _:
                    raise NotImplementedError(type(node))

            curr[node_id] = val

        curr, next_vals = next_vals, curr

    return {i: next_vals[i] for i in range(N)}


def evaluate_ltl_finite_dag_old(dag_nodes: list[DAGId], dag_root, T_pred: dict[str, np.ndarray], which=jnp):
    """Evaluate whether the LTL formula (when treated as finite) holds over the finite trace.
    Solve using dynamic programming."""

    tmp_key = list(T_pred.keys())[0]
    (T,) = T_pred[tmp_key].shape

    # 1. Obtain the final value.
    pred_final = {k: v[-1] for k, v in T_pred.items()}
    dag_values = {}
    get_values_old(dag_nodes, dag_root, pred_final, next_values=None, which=which, values=dag_values)
    assert len(dag_values) == len(dag_nodes)
    dag_values_curr = dag_values

    if which is np:
        # 2. Move backwards through time.
        for kk in range(T - 2, -1, -1):  # T-2, T-3, ..., 0
            pred = {k: v[kk] for k, v in T_pred.items()}

            dag_values_next = dag_values_curr
            dag_values_curr = {}
            get_values_old(dag_nodes, dag_root, pred, next_values=dag_values_next, which=np, values=dag_values_curr)
    else:
        # 2. Move backwards using scan.
        def step(dag_values_next, pred):
            dag_values_curr_ = {}
            get_values_old(dag_nodes, dag_root, pred, next_values=dag_values_next, which=which, values=dag_values_curr_)
            return dag_values_curr_, None

        T_pred_prefix_reversed = {k: v[:-1][::-1] for k, v in T_pred.items()}
        dag_values_curr, _ = jax.lax.scan(step, dag_values_curr, T_pred_prefix_reversed)

    return dag_values_curr


def get_values_old(
    dag_nodes: list[DAGNode],
    dag_root: DAGId,
    predicates: dict[str, np.ndarray],
    next_values: dict[DAGId, np.ndarray] | None,
    which=jnp,
    values: dict[DAGId, np.ndarray] | None = None,
    allow_const: bool = False,
):
    is_terminal = next_values is None
    get_values = get_values_old

    if values is None:
        values: dict[DAGId, np.ndarray] = {}

    if dag_root in values:
        return values[dag_root]

    node = dag_nodes[dag_root]
    match node:
        case DAGConst(value=value):
            if allow_const:
                val = which.full((), value)
            else:
                raise ValueError("Const should have been simplified away.")
        case DAGVar(name=name):
            val = predicates[name]
        case DAGNegate(arg=arg_id):
            arg_val = get_values(dag_nodes, arg_id, predicates, next_values, which, values)
            val = -arg_val
        case DAGMaxN(args=args_ids):
            vals = [get_values(dag_nodes, arg_id, predicates, next_values, which, values) for arg_id in args_ids]
            val = which.max(which.stack(vals, axis=0), axis=0)
        case DAGMinN(args=args_ids):
            vals = [get_values(dag_nodes, arg_id, predicates, next_values, which, values) for arg_id in args_ids]
            val = which.min(which.stack(vals, axis=0), axis=0)
        case DAGReach(reach=reach_id):
            reach_val = get_values(dag_nodes, reach_id, predicates, next_values, which, values)
            if is_terminal:
                # At the terminal step, Reach is just the value of the reach node.
                val = reach_val
            else:
                # reach OR next_values
                val = which.maximum(reach_val, next_values[dag_root])
        case DAGAvoid(avoid=stay_id):
            stay_val = get_values(dag_nodes, stay_id, predicates, next_values, which, values)
            if is_terminal:
                # At the terminal step, Avoid is just the value of the avoid node.
                val = stay_val
            else:
                # stay AND next_values
                val = which.minimum(stay_val, next_values[dag_root])
        case DAGReachAvoid(reach=reach_id, avoid=stay_id):
            reach_val = get_values(dag_nodes, reach_id, predicates, next_values, which, values)
            stay_val = get_values(dag_nodes, stay_id, predicates, next_values, which, values)
            if is_terminal:
                # At the terminal step, ReachAvoid is just the value of the reach node.
                val = reach_val
            else:
                # reach OR (stay AND next_values)
                val = which.maximum(reach_val, which.minimum(stay_val, next_values[dag_root]))

        case DAGGUMinN(args=args_ids):
            # Treat the GU Min as a normal Min
            vals = [get_values(dag_nodes, arg_id, predicates, next_values, which, values) for arg_id in args_ids]
            val = which.min(which.stack(vals, axis=0), axis=0)
        case DAGGUSingle(reach=reach_id, avoid=stay_id):
            # Treat a single GU as a normal ReachAvoid.
            reach_val = get_values(dag_nodes, reach_id, predicates, next_values, which, values)
            stay_val = get_values(dag_nodes, stay_id, predicates, next_values, which, values, allow_const=True)
            if is_terminal:
                # At the terminal step, ReachAvoid is just the value of the reach node.
                val = reach_val
            else:
                # reach OR (stay AND next_values)
                val = which.maximum(reach_val, which.minimum(stay_val, next_values[dag_root]))
        # case DAGGU(args=args_ids):
        #     raise ValueError("How to handle GU... ?")
        case _:
            raise NotImplementedError(f"Node type {type(node)} not implemented.")

    values[dag_root] = val
    return val


def eval_guard_condition(dag_nodes: list[DAGNode], dag_root: DAGId, predicates: dict[str, np.ndarray], which=jnp):
    # dag_root represents the reach part of either DAGReach, DAGReachAvoid or DAGGUSingle.
    # - If it is a max, then compute the guard condition of all the arguments of the max, and take the max.
    # - If it is a min, there should at most one temporal. Compute the value of all non-temporal, then take the min.
    # - If it is non-temporal, then compute the value.
    values = {}

    node = dag_nodes[dag_root]
    match node:
        case DAGMaxN(args=args_ids):
            guard_vals = [eval_guard_condition(dag_nodes, arg_id, predicates, which) for arg_id in args_ids]
            guard_val = which.max(which.stack(guard_vals, axis=0), axis=0)
            return guard_val
        case DAGMinN(args=args_ids):
            temporal_args = []
            nontemporal_args = []
            for arg_id in args_ids:
                if has_temporal_children(arg_id, dag_nodes, include_self=True):
                    temporal_args.append(arg_id)
                else:
                    nontemporal_args.append(arg_id)

            if len(temporal_args) > 1:
                raise ValueError("Multiple temporal arguments in a Min is not supported.")

            if len(nontemporal_args) == 0:
                raise ValueError(
                    "At least one non-temporal argument in a Min is required to compute the guard condition."
                )

            nontemporal_vals = [
                get_values_old(dag_nodes, arg_id, predicates, None, which, values=values) for arg_id in nontemporal_args
            ]
            guard_val = which.min(which.stack(nontemporal_vals, axis=0), axis=0)
            return guard_val
        case _:
            if has_temporal_children(dag_root, dag_nodes, include_self=True):
                raise ValueError("Temporal node with non-Max/Min root is not supported.")
            return get_values_old(dag_nodes, dag_root, predicates, None, which, values)
