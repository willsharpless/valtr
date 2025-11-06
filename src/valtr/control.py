import hj_reachability.dynamics as dynamics
import numpy as np
from hj_reachability import Grid
from scipy import integrate as ode

from valtr.reachability import DAGAvoid, DagBuilder, DAGId, DAGMaxN, DAGMinN, DAGReachAvoid


def model(t, x, grad_values, grid, dynamics, times=None, tv=False):
    # Time-varying
    if tv:
        assert times is not None
        i = np.argmin(np.abs(times - t))
        grad_value = grid.interpolate(grad_values[i], state=x)
    else:
        grad_value = grid.interpolate(grad_values, state=x)

    u = dynamics.optimal_control(x, t, grad_value)
    d = dynamics.optimal_disturbance(x, t, grad_value)
    fx = dynamics.open_loop_dynamics(x, t)
    Bu = dynamics.control_jacobian(x, t)
    Bd = dynamics.disturbance_jacobian(x, t)

    dx = fx + Bu @ u + Bd @ d
    dx = dx.tolist()
    return dx


def characteristic(t0, x0, grad_values, grid, dynamics, times=None, tv=False):
    sol = ode.solve_ivp(
        lambda t, x: model(t, x, grad_values, grid, dynamics, times=times, tv=tv), [t0, 0], x0, max_step=0.1
    )
    return sol


def construct_optimal_path(
    dag: DagBuilder,
    dag_values: dict[DAGId, np.ndarray],
    dag_grads: dict[DAGId, np.ndarray],
    t_start: float,
    x_start: np.ndarray,
    dag_id: DAGId,
    grid: Grid,
    dynamics: dynamics.Dynamics,
    times=None,
    tv=False,
    reaching_eps: float = 0.0,
):
    dag_nodes = dag.nodes

    print(
        "Rolling out - NODE [{}:{}] at t = {:2.2f}, x = {}".format(
            str(type(dag.nodes[dag_id])).split(".")[-1].split("'")[0], dag_id, t_start, x_start
        )
    )
    dag_path, switch_times = [dag_id], []
    grad_values = dag_grads[dag_id]
    sol = characteristic(t_start, x_start, grad_values, grid, dynamics, times=times, tv=tv)

    # (n_times, )
    sol_t: np.ndarray = sol.t
    # (nx, n_times)
    sol_y: np.ndarray = sol.y

    # Recursively overwrite sol until at an Avoid node (terminal)
    node = dag_nodes[dag_id]

    match node:
        case DAGAvoid(avoid=_):
            dag_path = [dag_id]
            switch_times = []
        case DAGReachAvoid(reach=reach_id, avoid=_):
            reach = dag.nodes[reach_id]

            # Handle multiple reach
            match reach:
                case DAGMaxN(args=args):
                    # Reach multiple AND
                    next_dag_ids = args
                case DAGMinN(args=_):
                    # Reach multiple OR
                    next_dag_ids = [reach_id]
                case _:
                    raise RuntimeError(
                        "Expected Min/Max node in DAGReachAvoid.reach but got: [{}:{}]".format(
                            str(type(reach)).split(".")[-1].split("'")[0], reach_id
                        )
                    )

            # ----------------------------------
            # Find the first index that we satisfy the reach condition.
            best_next_dag_id = None
            best_reach_index = np.inf

            for next_dag_id in next_dag_ids:
                values_next = dag_values[next_dag_id]
                sol_values = np.array([grid.interpolate(values_next, state=sol_y[:, i]) for i in range(len(sol_t))])
                # Find the first index where we satisfy the reach condition, if any.
                reach_index = np.argmax(sol_values > reaching_eps) if any(sol_values > reaching_eps) else np.inf

                if reach_index < best_reach_index:
                    best_reach_index = reach_index
                    best_next_dag_id = next_dag_id

            # ----------------------------------
            # If we satisfied the reach condition for a child, recurse
            if best_next_dag_id is not None:
                dag_path.append(reach_id)

                # Pick next RA/A problem #FIXME? more complex for (UNTIL) UNTIL (UNTIL) etc...
                next_node = dag.nodes[best_next_dag_id]
                match next_node:
                    case DAGAvoid() | DAGReachAvoid():
                        pass
                    case _:
                        dag_path.append(best_next_dag_id)  # will be overwritten
                        children_types = [type(dag.nodes[child]) for child in next_node.args]
                        if DAGReachAvoid in children_types:
                            best_next_dag_id = next_node.args[children_types.index(DAGReachAvoid)]
                        elif DAGAvoid in children_types:
                            best_next_dag_id = next_node.args[children_types.index(DAGAvoid)]
                        else:
                            raise RuntimeError(
                                "Could not find next ReachAvoid or Avoid node in children of node [{}:{}]".format(
                                    str(type(next_node)).split(".")[-1].split("'")[0], best_next_dag_id
                                )
                            )

                # if not (type(next_node) is DAGAvoid or type(next_node) is DAGReachAvoid):
                #     dag_path.append(best_next_dag_id)  # will be overwritten
                #     children_types = [type(dag.nodes[child]) for child in next_node.args]
                #     if DAGReachAvoid in children_types:
                #         best_next_dag_id = next_node.args[children_types.index(DAGReachAvoid)]
                #     elif DAGAvoid in children_types:
                #         best_next_dag_id = next_node.args[children_types.index(DAGAvoid)]
                #     else:
                #         raise RuntimeError(
                #             "Could not find next ReachAvoid or Avoid node in children of node [{}:{}]".format(
                #                 str(type(next_node)).split(".")[-1].split("'")[0], best_next_dag_id
                #             )
                #         )

                t_reach: float = sol_t[best_reach_index]
                x_reach = sol_y[:, best_reach_index]
                next_sol, next_dag_path, next_switch_times = construct_optimal_path(
                    dag,
                    dag_values,
                    dag_grads,
                    t_reach,
                    x_reach,
                    best_next_dag_id,
                    grid,
                    dynamics,
                    times=times,
                    tv=tv,
                    reaching_eps=reaching_eps,
                )

                # Combine solutions
                t_combined = np.concatenate([sol_t[: best_reach_index + 1], next_sol.t])
                x_combined = np.concatenate([sol_y[:, : best_reach_index + 1], next_sol.y], axis=1)
                sol = ode._ivp.ivp.OdeResult(t=t_combined, y=x_combined)
                dag_path = dag_path + next_dag_path
                switch_times = [t_reach] + next_switch_times

        case _:
            raise NotImplementedError("Expected either DAGAvoid or DAGReachAvoid")

    return sol, dag_path, switch_times

    # if isinstance(dag_nodes[dag_id], DAGAvoid):  # or type(dag_builder.nodes[dag_id]) is DAGReach:
    #     dag_path = [dag_id]
    #     switch_times = []
    #
    # # For each ReachAvoid node, find which child gives the earliest reaching value
    # else:
    #
    #     if type(dag.nodes[dag.nodes[dag_id].reach]) is DAGMaxN:
    #         next_dag_ids = dag.nodes[dag.nodes[dag_id].reach].args
    #     elif type(dag.nodes[dag.nodes[dag_id].reach]) is DAGMinN:
    #         next_dag_ids = [dag.nodes[dag_id].reach]
    #     else:
    #         raise RuntimeError(
    #             "Expected Min/Max node in DAGReachAvoid.reach but got: [{}:{}]".format(
    #                 str(type(dag.nodes[dag_id])).split(".")[-1].split("'")[0], dag_id
    #             )
    #         )
    #     best_next_dag_id = None
    #     best_reach_index = np.inf
    #
    #     for next_dag_id in next_dag_ids:
    #         values_next = dag_values[next_dag_id]
    #         sol_values = np.array([grid.interpolate(values_next, state=sol.y[:, i]) for i in range(len(sol.t))])
    #         reach_index = np.argmax(sol_values > reaching_eps) if any(sol_values > reaching_eps) else np.inf
    #
    #         if reach_index < best_reach_index:
    #             best_reach_index = reach_index
    #             best_next_dag_id = next_dag_id
    #
    #     # If reaching child, recurse
    #     if best_next_dag_id is not None:
    #         dag_path.append(dag.nodes[dag_id].reach)
    #
    #         # Pick next RA/A problem #FIXME? more complex for (UNTIL) UNTIL (UNTIL) etc...
    #         next_node = dag.nodes[best_next_dag_id]
    #         if not (type(next_node) is DAGAvoid or type(next_node) is DAGReachAvoid):
    #             dag_path.append(best_next_dag_id)  # will be overwritten
    #             children_types = [type(dag.nodes[child]) for child in next_node.args]
    #             if DAGReachAvoid in children_types:
    #                 best_next_dag_id = next_node.args[children_types.index(DAGReachAvoid)]
    #             elif DAGAvoid in children_types:
    #                 best_next_dag_id = next_node.args[children_types.index(DAGAvoid)]
    #             else:
    #                 raise RuntimeError(
    #                     "Could not find next ReachAvoid or Avoid node in children of node [{}:{}]".format(
    #                         str(type(next_node)).split(".")[-1].split("'")[0], best_next_dag_id
    #                     )
    #                 )
    #
    #         t_reach = sol.t[best_reach_index]
    #         x_reach = sol.y[:, best_reach_index]
    #         next_sol, next_dag_path, next_switch_times = construct_optimal_path(
    #             dag,
    #             dag_values,
    #             dag_grads,
    #             t_reach,
    #             x_reach,
    #             best_next_dag_id,
    #             grid,
    #             dynamics,
    #             times=times,
    #             tv=tv,
    #             reaching_eps=reaching_eps,
    #         )
    #
    #         # Combine solutions
    #         t_combined = np.concatenate([sol.t[: best_reach_index + 1], next_sol.t])
    #         x_combined = np.concatenate([sol.y[:, : best_reach_index + 1], next_sol.y], axis=1)
    #         sol = ode._ivp.ivp.OdeResult(t=t_combined, y=x_combined)
    #         dag_path = dag_path + next_dag_path
    #         switch_times = [t_reach] + next_switch_times
    #
    #     # Else (eg. no more time, impossible reach), return the current RA solution
    #
    # return sol, dag_path, switch_times
