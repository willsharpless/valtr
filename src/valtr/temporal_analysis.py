import ipdb
import tqdm
import jax
import jax.numpy as jnp
import numpy as np

from valtr.reachability import (DAGAvoid, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGReach, DAGReachAvoid, DAGVar)


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
