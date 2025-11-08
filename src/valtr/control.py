from turtle import mode
import hj_reachability.dynamics as dynamics
import numpy as np
from hj_reachability import Grid
from scipy import integrate as ode
import matplotlib.pyplot as plt
from valtr.reachability import DAGAvoid, DagBuilder, DAGId, DAGMaxN, DAGMinN, DAGReachAvoid
import faster_hj_grid_interpolation # patches on faster itp
import copy
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Tuple, Any
import functools

def rk4_step(model, t, y, h):
    k1 = h * model(t, y)
    k2 = h * model(t + h/2, y + k1/2)
    k3 = h * model(t + h/2, y + k2/2)
    k4 = h * model(t + h, y + k3)
    
    y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    return y_new

def jax_ivp(t0, x0, grad_values, grid, dynamics, times=None, tv=False, step_size=0.01, max_steps=10000):
    def jax_model(t, x):
        # t_np, x_np = np.array(t), np.array(x)
        t_np, x_np = t, x

        if tv:
            i = np.argmin(np.abs(times - t_np))
            grad_value = grid.interpolate_fast_jit(grad_values[i], state=x_np)
        else:
            grad_value = grid.interpolate_fast_jit(grad_values, state=x_np)
        
        u = dynamics.optimal_control(x_np, t_np, grad_value)
        d = dynamics.optimal_disturbance(x_np, t_np, grad_value)
        fx = dynamics.open_loop_dynamics(x_np, t_np)
        Bu = dynamics.control_jacobian(x_np, t_np)
        Bd = dynamics.disturbance_jacobian(x_np, t_np)
        
        dx = fx + Bu @ u + Bd @ d
        # dx = dx.tolist()
        # return jnp.array(dx)
        return dx
    
    return jax_solve_ivp_no_jit(
        jax_model, 
        [t0, 0], 
        x0, 
        step_size=step_size,
        max_steps=max_steps
    )

def jax_solve_ivp_no_jit(
    model,
    t_span: Tuple[float, float],
    y0: jnp.ndarray,
    step_size: float = 0.01,
    max_steps: int = 10000
):
    t0, tf = t_span
    dt = step_size
    t_history = [t0]
    y_history = [y0]
    n_steps = min(int(np.ceil(abs(tf - t0) / abs(dt))), max_steps)

    t, y = t0, y0
    for _ in range(n_steps):
        y = rk4_step(model, t, y, dt)
        t = t + dt
        t_history.append(t)
        y_history.append(y)
    
    t_result = jnp.array(t_history)
    y_result = jnp.stack(y_history, axis=0)
    
    return {
        't': t_result,
        'y': y_result.T,  # Match scipy format
        'success': True,
        'nfev': len(t_history) * 4  # RK4 uses 4 evaluations per step
    }
    
def model(t, x, grad_values, grid, dynamics, times=None, tv=False):
    if tv:
        assert times is not None
        i = np.argmin(np.abs(times - t))
        grad_value = grid.interpolate_fast_jit(grad_values[i], state=x)
    else:
        grad_value = grid.interpolate_fast_jit(grad_values, state=x)

    u = dynamics.optimal_control(x, t, grad_value)
    d = dynamics.optimal_disturbance(x, t, grad_value)
    fx = dynamics.open_loop_dynamics(x, t)
    Bu = dynamics.control_jacobian(x, t)
    Bd = dynamics.disturbance_jacobian(x, t)

    dx = fx + Bu @ u + Bd @ d
    dx = dx.tolist()
    return dx

def characteristic(t0, x0, grad_values, grid, dynamics, times=None, tv=False, method='jax'):
    if method == 'jax':
        sol = jax_ivp(t0, x0, grad_values, grid, dynamics, times=times, tv=tv, step_size=0.01, max_steps=10000)
        sol = ode._ivp.ivp.OdeResult(t=sol['t'], y=sol['y'])
    elif method == 'scipy':
        sol = ode.solve_ivp(
            lambda t, x: model(t, x, grad_values, grid, dynamics, times=times, tv=tv), [t0, 0], x0, max_step=0.01, atol=1., rtol=1.,  # (makes fixed step size)
        )
    else:
        raise ValueError(f"Unknown method '{method}' for characteristic integration.")
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
    integration_method: str = 'jax',
):
    dag_nodes = dag.nodes

    print(
        "Rolling out - NODE [{}:{}] at t = {:2.2f}, x = {}".format(
            str(type(dag.nodes[dag_id])).split(".")[-1].split("'")[0], dag_id, t_start, x_start
        )
    )
    dag_path, switch_times = [dag_id], []
    grad_values = dag_grads[dag_id]
    
    # Use JAX or scipy integration
    sol = characteristic(t_start, x_start, grad_values, grid, dynamics, times=times, tv=tv, method=integration_method)

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
                sol_values = grid.interpolate_fast_batch_jit(values_next, states=sol_y.T)

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
                    integration_method=integration_method,  # Pass the new parameter down
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

def plot_optimal_path(sol, dag_ra_path, switch_times, fig_base=None):
    
    if fig_base is None:
        fig, ax = plt.subplots()
    else:
        fig = copy.deepcopy(fig_base)
        ax = fig.axes[0]

    ax.plot(sol.y[0,:], sol.y[1,:], 'k-', linewidth=2, label='Optimal Path')
    ax.plot(sol.y[0,0], sol.y[1,0], 'o', markersize=6, label='', color='white')
    ax.plot(sol.y[0,0], sol.y[1,0], 'x', markersize=5, label='Start', color='black')
    
    dag_path_c = 1
    for st in switch_times:
        switch_index = np.argmin(np.abs(sol.t - st))
        tab_color = plt.get_cmap('tab10')(dag_path_c % 10)
        ax.plot(sol.y[0,switch_index], sol.y[1,switch_index], 'o', markersize=6, label='switch: %d'%(dag_ra_path[dag_path_c]), color=tab_color, markeredgecolor='black', markeredgewidth=1)
        ax.text(sol.y[0,switch_index] + 0.05, sol.y[1,switch_index] + 0.02, "{:d}→{:d}".format(dag_ra_path[dag_path_c-1], dag_ra_path[dag_path_c]), color='black', fontsize=8)
        dag_path_c += 1

    ax.legend()
    ax.set_title("Value Optimal Path")
    fig.savefig("sol_path.pdf", bbox_inches="tight")
    return fig