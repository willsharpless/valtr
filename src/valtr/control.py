from turtle import mode
import hj_reachability.dynamics as dynamics
import numpy as np
from hj_reachability import Grid
from scipy import integrate as ode
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from valtr.reachability import DAGAvoid, DAGConst, DAGVar, DagBuilder, DAGId, DAGMaxN, DAGMinN, DAGReachAvoid
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

    # Recursively overwrite sol until at an Avoid node (terminal) or a terminal ReachAvoid node (reach=var/const)
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
                case DAGVar() | DAGConst():
                    # terminal reachavoid
                    dag_path = [dag_id]
                    switch_times = []
                    # shorten reach-avoid path to first reach satisfaction
                    reach_point = grid.interpolate_fast_batch_jit(dag_values[reach_id], states=sol_y.T)
                    reach_indices = np.where(reach_point > reaching_eps)[0]
                    if len(reach_indices) > 0:
                        first_reach_index = reach_indices[0]
                        sol_t = sol_t[: first_reach_index + 1]
                        sol_y = sol_y[:, : first_reach_index + 1]
                        sol = ode._ivp.ivp.OdeResult(t=sol_t, y=sol_y)
                        print("  Terminal reach-avoid satisfied at t = {:2.2f}, ending early".format(sol_t[first_reach_index]))
                    return sol, dag_path, switch_times
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

def plot_optimal_path(sol, dag_ra_path, switch_times, fig_base=None, 
                      color_path=False, color_path_type='state', color_path_state_ix=-1, 
                      color_path_lb=None, color_path_ub=None, color_path_state_label=''):
    
    if fig_base is None:
        fig, ax = plt.subplots()
    else:
        fig = copy.deepcopy(fig_base)
        ax = fig.axes[0]

    if not color_path:
        ax.plot(sol.y[0,:], sol.y[1,:], 'k-', linewidth=2, label='Optimal Path')
        ax.plot(sol.y[0,0], sol.y[1,0], 'o', markersize=6, label='', color='white')
        ax.plot(sol.y[0,0], sol.y[1,0], 'x', markersize=5, label='Start', color='black')
    else:
        color_map = plt.get_cmap('RdYlGn')
        if color_path_type == 'state':
            color_traj = sol.y[color_path_state_ix,:]
            color_path_lb = jnp.min(color_traj) if color_path_lb is None else color_path_lb
            color_path_ub = jnp.max(color_traj) if color_path_ub is None else color_path_ub
            norm = plt.Normalize(vmin=color_path_lb, vmax=color_path_ub)
        else:
            color_traj = sol.t
        # ax.scatter(sol.y[0,:], sol.y[1,:], 'k-', linewidth=2, label='Optimal Path', color=color_map(norm(color_traj)))
        
        # Create line segments
        points = np.array([sol.y[0,:], sol.y[1,:]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create LineCollection with color mapping
        lc = LineCollection(segments, cmap=color_map, norm=norm, linewidth=2, label='Optimal Path')
        lc.set_array(color_traj)
        line = ax.add_collection(lc)        

        ax.plot(sol.y[0,0], sol.y[1,0], 'o', markersize=6, label='', color='white')
        ax.plot(sol.y[0,0], sol.y[1,0], 'x', markersize=5, label='Start', color='black')
        # # Add colorbar
        # sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        # sm.set_array([])
        cbar = fig.colorbar(line, ax=ax)
        if color_path_type == 'state':
            cbar.set_label(color_path_state_label, fontsize=8)

    dag_path_c = 1
    for st in switch_times:
        switch_index = np.argmin(np.abs(sol.t - st))
        tab_color = plt.get_cmap('tab10')(dag_path_c % 10)
        ax.plot(sol.y[0,switch_index], sol.y[1,switch_index], 'o', markersize=6, label='switch: %d'%(dag_ra_path[dag_path_c]), color=tab_color, markeredgecolor='black', markeredgewidth=1)
        ax.text(sol.y[0,switch_index] + 0.05, sol.y[1,switch_index] + 0.02, "{:d}→{:d}".format(dag_ra_path[dag_path_c-1], dag_ra_path[dag_path_c]), color='black', fontsize=8)
        dag_path_c += 1

    ax.legend(frameon=True, facecolor="white", framealpha=0.8)
    ax.set_title("Value - Optimal Path")
    fig.savefig("solution_path.pdf", bbox_inches="tight")
    return fig

# Batch versions (non recursive)

"""
Batched Optimal Path Construction

This module provides high-performance batched versions of optimal path construction
for processing multiple initial conditions in parallel using JAX.

Key Functions:
- construct_optimal_path_batch(): General batched implementation
- construct_optimal_path_batch_fast(): Highly optimized JAX scan-based version  
- construct_optimal_path_batch_auto(): Convenience function with automatic preprocessing

Usage Example:
    # Prepare batch of initial conditions
    X_start = jnp.array([[x1, y1], [x2, y2], [x3, y3], ...])  # (batch_size, state_dim)
    
    # Run batched path construction
    results = construct_optimal_path_batch_auto(
        dag, dag_values, dag_grads, t_start, X_start, dag_root_id,
        grid, dynamics, use_fast_version=True
    )
    
    # Extract individual trajectories
    for i in range(batch_size):
        sol, dag_path, switch_times = extract_single_trajectory(results, i)

Performance Notes:
- The fast version uses JAX scan operations for maximum throughput
- All operations are JIT compiled for minimal overhead
- Memory usage scales linearly with batch size
- Recommended for batch sizes > 100 initial conditions
"""

def construct_optimal_path_batch(
    dag: DagBuilder,
    dag_values: dict[DAGId, np.ndarray],
    dag_grads: dict[DAGId, np.ndarray],
    t_start: float,
    X_start: np.ndarray,
    dag_root_id: DAGId,
    grid: Grid,
    dynamics: dynamics.Dynamics,
    times=None,
    tv=False,
    reaching_eps: float = 0.0,
    integration_method: str = 'jax',
    max_switches: int = 10,
    max_integration_time: float = 10.0,
    step_size: float = 0.01
):
    """
    Efficient vectorized batch implementation that groups trajectories by DAG state.
    
    This avoids JAX tracing issues by processing trajectories in groups that share the same DAG node,
    allowing for true vectorization of the expensive integration steps.
    """
    X_start = np.array(X_start)
    batch_size = X_start.shape[0]
    state_dim = X_start.shape[1]
    
    # Initialize tracking arrays
    current_states = X_start.astype(np.float32)
    current_times = np.full(batch_size, t_start, dtype=np.float32)
    current_dag_ids = np.full(batch_size, dag_root_id, dtype=np.int32)
    active_mask = np.ones(batch_size, dtype=bool)
    
    # Results storage
    max_path_length = max_switches * 3 + 1  # Allow for intermediate nodes
    dag_paths = np.full((batch_size, max_path_length), -1, dtype=np.int32)
    dag_paths[:, 0] = dag_root_id
    switch_times_array = np.full((batch_size, max_switches), np.inf, dtype=np.float32)
    path_lengths = np.ones(batch_size, dtype=np.int32)
    
    # Trajectory history for GIF creation
    max_history_steps = int(max_integration_time / step_size) + 100  # Buffer for safety
    trajectory_history = {
        'states': np.full((batch_size, max_history_steps, state_dim), np.nan, dtype=np.float32),
        'times': np.full((batch_size, max_history_steps), np.nan, dtype=np.float32),
        'dag_ids': np.full((batch_size, max_history_steps), -1, dtype=np.int32),
        'step_count': 0
    }
    
    def record_trajectory_step():
        """Record current state in trajectory history"""
        step = trajectory_history['step_count']
        if step < max_history_steps:
            trajectory_history['states'][:, step] = current_states
            trajectory_history['times'][:, step] = current_times
            trajectory_history['dag_ids'][:, step] = current_dag_ids
            trajectory_history['step_count'] += 1
    
    # Pre-compile JAX functions for different DAG nodes to avoid dynamic indexing
    def create_dynamics_function(dag_id):
        """Create a JIT-compiled dynamics function for a specific DAG node"""
        grad_values = dag_grads[dag_id]
        
        @jax.jit
        def dynamics_for_dag_id(states, times):
            def single_dynamics(state, time):
                if tv and times is not None:
                    time_idx = jnp.argmin(jnp.abs(times - time))
                    grad_value = grid.interpolate_fast_jit(grad_values[time_idx], state=state)
                else:
                    grad_value = grid.interpolate_fast_jit(grad_values, state=state)
                
                u = dynamics.optimal_control(state, time, grad_value)
                d = dynamics.optimal_disturbance(state, time, grad_value)
                fx = dynamics.open_loop_dynamics(state, time)
                Bu = dynamics.control_jacobian(state, time)
                Bd = dynamics.disturbance_jacobian(state, time)
                
                return fx + Bu @ u + Bd @ d
            
            return jax.vmap(single_dynamics)(states, times)
        
        return dynamics_for_dag_id
    
    # Pre-compile dynamics functions for all DAG nodes that have gradients
    dynamics_functions = {}
    for dag_id in dag_grads.keys():
        dynamics_functions[dag_id] = create_dynamics_function(dag_id)
    
    def check_reach_conditions(states, dag_ids_array, active):
        """Non-JIT reach condition checking to avoid DAG indexing issues"""
        should_switch = np.zeros(batch_size, dtype=bool)
        next_dag_ids = dag_ids_array.copy()
        intermediate_nodes = []  # Track intermediate nodes for path reconstruction
        
        for i in range(batch_size):
            if not active[i]:
                intermediate_nodes.append([])
                continue
            
            state = states[i]
            dag_id = dag_ids_array[i]
            node = dag.nodes[dag_id]
            
            if isinstance(node, DAGAvoid):
                # Terminal node - don't switch, just stay on this DAG node
                should_switch[i] = False
                next_dag_ids[i] = dag_id  # Stay on the same terminal node
                intermediate_nodes.append([])
            
            elif isinstance(node, DAGReachAvoid) and \
               (isinstance(dag.nodes[node.reach], (DAGVar, DAGConst))):
                # Terminal node - don't switch, just stay on this DAG node
                should_switch[i] = False
                next_dag_ids[i] = dag_id  # Stay on the same terminal node
                intermediate_nodes.append([])

                # check if reached and deactivate
                reach_id = node.reach
                values_reach = dag_values[reach_id]
                state_value = grid.interpolate_fast_jit(values_reach, state=state)
                if state_value > reaching_eps:
                    active[i] = False

            elif isinstance(node, DAGReachAvoid):
                reach_id = node.reach
                reach_node = dag.nodes[reach_id]
                
                # Get candidates
                if isinstance(reach_node, DAGMaxN):
                    candidates = reach_node.args
                elif isinstance(reach_node, DAGMinN):
                    candidates = [reach_id]
                else:
                    candidates = [reach_id]
                
                best_next_id = -1
                best_value = -np.inf
                
                for next_id in candidates:
                    if next_id in dag_values:
                        values_next = dag_values[next_id]
                        state_value = grid.interpolate_fast_jit(values_next, state=state)
                        
                        if state_value > reaching_eps and state_value > best_value:
                            best_value = state_value
                            best_next_id = next_id
                
                if best_next_id != -1:
                    # Track intermediate nodes like the single version
                    path_intermediates = [reach_id]  # Always add reach node
                    
                    # Handle composite nodes
                    next_node = dag.nodes[best_next_id]
                    if not isinstance(next_node, (DAGAvoid, DAGReachAvoid)):
                        # Composite node - add it to path and find actual child
                        path_intermediates.append(best_next_id)
                        
                        # Find the RA/A child like the single version does
                        children_types = [type(dag.nodes[child]) for child in next_node.args]
                        if DAGReachAvoid in children_types:
                            best_next_dag_id = next_node.args[children_types.index(DAGReachAvoid)]
                        elif DAGAvoid in children_types:
                            best_next_dag_id = next_node.args[children_types.index(DAGAvoid)]
                        else:
                            # Fallback - just use the composite node ID
                            best_next_dag_id = best_next_id
                    else:
                        # Direct RA/A node
                        best_next_dag_id = best_next_id
                    
                    should_switch[i] = True
                    next_dag_ids[i] = best_next_dag_id
                    intermediate_nodes.append(path_intermediates)
                else:
                    intermediate_nodes.append([])
            else:
                intermediate_nodes.append([])
        
        return should_switch, next_dag_ids, intermediate_nodes, active
    
    # Main integration loop - each trajectory evolves independently
    target_time = 0.0  # Integration target time
    max_total_steps = int(abs(target_time - t_start) / step_size)
    
    for global_step in range(max_total_steps):
        # Record trajectory state
        record_trajectory_step()
        
        # Progress reporting
        if global_step % 100 == 0:
            min_time = np.min(current_times[active_mask]) if np.any(active_mask) else target_time
            max_time = np.max(current_times[active_mask]) if np.any(active_mask) else target_time
        
        # Note: Don't check for trajectory completion - continue until full time horizon
            
        # Check reach conditions for all active trajectories
        should_switch, next_dag_ids, intermediate_nodes, active_mask = check_reach_conditions(
            current_states, current_dag_ids, active_mask
        )
        
        # Handle switching for each trajectory independently
        for i in range(batch_size):
            if active_mask[i] and should_switch[i]:
                # Count the number of switches for this trajectory
                trajectory_switches = np.sum(switch_times_array[i] < np.inf)
                
                if trajectory_switches < max_switches:
                    # Add intermediate nodes to path
                    for intermediate_id in intermediate_nodes[i]:
                        if path_lengths[i] < max_path_length:
                            dag_paths[i, path_lengths[i]] = intermediate_id
                            path_lengths[i] += 1
                    
                    # Add final target node
                    if path_lengths[i] < max_path_length:
                        dag_paths[i, path_lengths[i]] = next_dag_ids[i]
                        path_lengths[i] += 1
                    
                    # Record switch time
                    switch_times_array[i, trajectory_switches] = current_times[i]
                    
                    # Update DAG ID
                    current_dag_ids[i] = next_dag_ids[i]
                    
                    # Note: Don't deactivate on terminal Avoid nodes - keep integrating
                    # Trajectories continue until time horizon is reached
        
        # Integrate all active trajectories grouped by DAG ID
        if np.any(active_mask):
            # Group trajectories by current DAG ID
            unique_dag_ids = np.unique(current_dag_ids[active_mask])
            
            for dag_id in unique_dag_ids:
                if dag_id not in dynamics_functions:
                    continue
                
                # Get trajectories with this DAG ID
                dag_mask = active_mask & (current_dag_ids == dag_id)
                if not np.any(dag_mask):
                    continue
                
                # Extract group data
                group_indices = np.where(dag_mask)[0]
                group_states = jnp.array(current_states[dag_mask])
                group_times = jnp.array(current_times[dag_mask])
                
                # Vectorized integration for this DAG group
                dynamics_func = dynamics_functions[dag_id]
                dx_group = dynamics_func(group_states, group_times)
                
                # Use fixed step size for consistent timing
                new_states = group_states + step_size * dx_group
                new_times = group_times + step_size
                
                # Write back to main arrays
                current_states[group_indices] = np.array(new_states)
                current_times[group_indices] = np.array(new_times)
    
    # Record final state
    record_trajectory_step()
    
    # Trim trajectory history to actual length
    actual_steps = trajectory_history['step_count']
    trajectory_history['states'] = trajectory_history['states'][:, :actual_steps]
    trajectory_history['times'] = trajectory_history['times'][:, :actual_steps]
    trajectory_history['dag_ids'] = trajectory_history['dag_ids'][:, :actual_steps]
    
    # Package results
    results = {
        'dag_paths': dag_paths,
        'switch_times': switch_times_array,
        'path_lengths': path_lengths,
        'final_states': current_states,
        'final_times': current_times,
        'active_mask': active_mask,
        'trajectory_lengths': np.full(batch_size, max_total_steps),
        'trajectory_history': trajectory_history
    }
    
    return results

def extract_single_trajectory(batch_results: dict, trajectory_idx: int) -> Tuple[Any, list, list]:
    """
    Extract a single trajectory from batched results in format compatible with original function.
    
    Args:
        batch_results: Results from construct_optimal_path_batch
        trajectory_idx: Index of trajectory to extract
        
    Returns:
        sol: OdeResult-like object with trajectory
        dag_path: List of DAG node IDs visited  
        switch_times: List of switch times
    """
    # Extract path for this trajectory
    path_length = batch_results['path_lengths'][trajectory_idx]
    dag_path = batch_results['dag_paths'][trajectory_idx, :path_length].tolist()
    
    # Extract switch times (filter out inf values)
    switch_times_raw = batch_results['switch_times'][trajectory_idx]
    switch_times = switch_times_raw[switch_times_raw < jnp.inf].tolist()
    
    # Extract trajectory
    traj_length = batch_results['trajectory_lengths'][trajectory_idx]
    t_traj = batch_results['trajectories_t'][trajectory_idx, :traj_length]
    x_traj = batch_results['trajectories_x'][trajectory_idx, :traj_length, :].T
    
    # Create scipy-compatible result object
    sol = ode._ivp.ivp.OdeResult(t=np.array(t_traj), y=np.array(x_traj))
    
    return sol, dag_path, switch_times

# Alternative fast implementation using pure JAX scan operations
def construct_optimal_path_batch_fast(
    dag_values_array: jnp.ndarray,  # Pre-stacked values for all DAG nodes 
    dag_grads_array: jnp.ndarray,   # Pre-stacked gradients for all DAG nodes
    node_types: jnp.ndarray,        # Array encoding node types (0=Avoid, 1=ReachAvoid, etc.)
    node_children: jnp.ndarray,     # Array encoding child relationships
    t_start: float,
    X_start: jnp.ndarray,
    dag_root_id: int,
    grid: Grid,                     # Grid object for interpolation
    dynamics: dynamics.Dynamics,    # Dynamics object
    reaching_eps: float = 0.0,
    max_switches: int = 5,
    max_steps: int = 1000,
    step_size: float = 0.01
) -> dict:
    """
    Highly optimized JAX implementation using scan operations.
    
    This version pre-processes the DAG into arrays for efficient JAX operations
    and uses scan for the main integration loop.
    """
    batch_size = X_start.shape[0]
    state_dim = X_start.shape[1]
    
    # JIT compile the inner functions for maximum performance
    @jax.jit
    def integration_step(carry, _):
        """Single step of trajectory integration with switching logic"""
        (states, times, dag_ids, paths, switch_times, path_lens, active) = carry
        
        # Compute dynamics for all active trajectories
        def compute_dynamics(state, time, dag_id, active_flag):
            # Use conditional to avoid computation for inactive trajectories
            return jax.lax.cond(
                active_flag,
                lambda: compute_dynamics_active(state, time, dag_id),
                lambda: jnp.zeros_like(state)
            )
        
        def compute_dynamics_active(state, time, dag_id):
            # Get gradients for current DAG node - use dynamic indexing carefully
            grad_values = dag_grads_array[dag_id]
            grad_value = grid.interpolate_fast_jit(grad_values, state=state)
            
            # Compute optimal control and dynamics
            u = dynamics.optimal_control(state, time, grad_value) 
            d = dynamics.optimal_disturbance(state, time, grad_value)
            fx = dynamics.open_loop_dynamics(state, time)
            Bu = dynamics.control_jacobian(state, time)
            Bd = dynamics.disturbance_jacobian(state, time)
            
            return fx + Bu @ u + Bd @ d
        
        # Vectorized dynamics computation
        dynamics_batch = jax.vmap(compute_dynamics)(states, times, dag_ids, active)
        
        # Update states and times
        new_states = states + step_size * dynamics_batch
        new_times = times + step_size
        
        # Check switching conditions
        def check_switch(state, dag_id, active_flag):
            return jax.lax.cond(
                active_flag,
                lambda: check_switch_active(state, dag_id),
                lambda: (False, dag_id)
            )
        
        def check_switch_active(state, dag_id):
            node_type = node_types[dag_id]
            
            # Terminal node (Avoid)
            terminal_case = lambda: (True, dag_id)
            
            # ReachAvoid node  
            def reachavoid_case():
                children = node_children[dag_id]
                best_child = dag_id
                best_value = -jnp.inf
                
                # Check all children (fixed size loop for JAX)
                def check_child(i, carry):
                    best_child, best_value = carry
                    child_id = children[i]
                    
                    # Skip invalid children
                    valid_child = child_id != -1
                    
                    def process_valid_child():
                        values = dag_values_array[child_id]
                        value = grid.interpolate_fast_jit(values, state=state)
                        better = (value > reaching_eps) & (value > best_value)
                        return jax.lax.cond(
                            better,
                            lambda: (child_id, value),
                            lambda: (best_child, best_value)
                        )
                    
                    return jax.lax.cond(
                        valid_child,
                        process_valid_child,
                        lambda: (best_child, best_value)
                    )
                
                # Assuming max 4 children - adjust as needed
                best_child, best_value = jax.lax.fori_loop(
                    0, 4, check_child, (dag_id, -jnp.inf)
                )
                
                switched = best_child != dag_id
                return switched, best_child
            
            return jax.lax.cond(
                node_type == 0,
                terminal_case,
                reachavoid_case
            )
        
        # Vectorized switching check
        switch_flags, new_dag_ids = jax.vmap(check_switch)(new_states, dag_ids, active)
        
        # Update paths and switch times where switching occurred
        switching_and_active = switch_flags & active
        
        # Safely update paths using scatter operations
        indices = jnp.where(
            switching_and_active, 
            path_lens, 
            -1  # Invalid index that won't update anything
        )
        
        # Only update where indices are valid
        valid_updates = indices >= 0
        safe_indices = jnp.where(valid_updates, indices, 0)
        safe_values = jnp.where(valid_updates, new_dag_ids, paths[:, 0])  # Use existing value if invalid
        
        new_paths = paths.at[jnp.arange(batch_size), safe_indices].set(
            safe_values, mode='fill', indices_are_sorted=False
        )
        
        # Similar safe update for switch times
        switch_indices = jnp.where(switching_and_active, path_lens - 1, -1)
        safe_switch_indices = jnp.where(switch_indices >= 0, switch_indices, 0)
        safe_switch_values = jnp.where(switch_indices >= 0, new_times, switch_times[:, 0])
        
        new_switch_times = switch_times.at[jnp.arange(batch_size), safe_switch_indices].set(
            safe_switch_values, mode='fill'
        )
        
        new_path_lens = jnp.where(switching_and_active, path_lens + 1, path_lens)
        
        # Update active status (deactivate if reached terminal)
        terminal_reached = switch_flags & (node_types[new_dag_ids] == 0)
        new_active = active & ~terminal_reached
        
        new_carry = (new_states, new_times, new_dag_ids, new_paths, 
                    new_switch_times, new_path_lens, new_active)
        
        # Return trajectory point for history
        trajectory_point = (new_states, new_times)
        
        return new_carry, trajectory_point
    
    # Initialize state
    initial_paths = jnp.full((batch_size, max_switches + 1), -1)
    initial_paths = initial_paths.at[:, 0].set(dag_root_id)
    initial_switch_times = jnp.full((batch_size, max_switches), jnp.inf)
    initial_active = jnp.ones(batch_size, dtype=bool)
    initial_path_lens = jnp.ones(batch_size, dtype=jnp.int32)
    
    initial_carry = (
        X_start, 
        jnp.full(batch_size, t_start),
        jnp.full(batch_size, dag_root_id, dtype=jnp.int32),
        initial_paths,
        initial_switch_times, 
        initial_path_lens,
        initial_active
    )
    
    # Run main integration loop
    final_carry, trajectory_history = jax.lax.scan(
        integration_step,
        initial_carry, 
        jnp.arange(max_steps)
    )
    
    # Unpack results
    (final_states, final_times, final_dag_ids, final_paths, 
     final_switch_times, final_path_lens, final_active) = final_carry
    
    traj_states, traj_times = trajectory_history
    
    return {
        'final_states': final_states,
        'final_times': final_times,
        'final_dag_ids': final_dag_ids,
        'dag_paths': final_paths,
        'switch_times': final_switch_times,
        'path_lengths': final_path_lens,
        'active_mask': final_active,
        'trajectory_states': traj_states.transpose((1, 0, 2)),  # (batch, time, state_dim)
        'trajectory_times': traj_times.T,  # (batch, time)
    }

def preprocess_dag_for_batch(
    dag: DagBuilder,
    dag_values: dict[DAGId, np.ndarray], 
    dag_grads: dict[DAGId, np.ndarray],
    max_children: int = 4
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert DAG structure to arrays for efficient JAX batch processing.
    
    Returns:
        dag_values_array: (num_nodes, *value_shape) values for each node
        dag_grads_array: (num_nodes, *grad_shape) gradients for each node  
        node_types: (num_nodes,) type encoding (0=Avoid, 1=ReachAvoid)
        node_children: (num_nodes, max_children) child IDs (-1 for invalid)
    """
    nodes = dag.nodes
    num_nodes = len(nodes)
    
    # Get shapes from first value/grad
    first_key = next(iter(dag_values.keys()))
    value_shape = dag_values[first_key].shape
    grad_shape = dag_grads[first_key].shape
    
    # Initialize arrays
    dag_values_array = np.zeros((num_nodes,) + value_shape)
    dag_grads_array = np.zeros((num_nodes,) + grad_shape)
    node_types = np.full(num_nodes, -1, dtype=np.int32)  # -1 = invalid
    node_children = np.full((num_nodes, max_children), -1, dtype=np.int32)
    
    # Fill arrays
    for node_id, node in enumerate(dag.nodes):
        if node_id in dag_values:
            dag_values_array[node_id] = dag_values[node_id]
            dag_grads_array[node_id] = dag_grads[node_id]
        
        # Encode node type
        if isinstance(node, DAGAvoid):
            node_types[node_id] = 0
        elif isinstance(node, DAGReachAvoid):
            node_types[node_id] = 1
            
            # Get children from reach node
            reach_node = dag.nodes[node.reach]
            if isinstance(reach_node, (DAGMaxN, DAGMinN)):
                children = reach_node.args[:max_children]  # Truncate if too many
                node_children[node_id, :len(children)] = children
        elif isinstance(node, (DAGMaxN, DAGMinN)):
            node_types[node_id] = 2  # Composite node
            children = node.args[:max_children]
            node_children[node_id, :len(children)] = children
    
    return (
        jnp.array(dag_values_array),
        jnp.array(dag_grads_array), 
        jnp.array(node_types),
        jnp.array(node_children)
    )

# Convenience function that combines preprocessing and batch execution
def construct_optimal_path_batch_auto(
    dag: DagBuilder,
    dag_values: dict[DAGId, np.ndarray],
    dag_grads: dict[DAGId, np.ndarray],
    t_start: float,
    X_start: np.ndarray,
    dag_root_id: DAGId,
    grid: Grid,
    dynamics: dynamics.Dynamics,
    use_fast_version: bool = True,
    **kwargs
) -> dict:
    """
    Automatically choose and execute the appropriate batched path construction.
    
    Args:
        use_fast_version: If True, use the highly optimized JAX scan implementation
        **kwargs: Additional arguments passed to the chosen implementation
    """
    if use_fast_version:
        # Preprocess DAG structure
        dag_values_array, dag_grads_array, node_types, node_children = preprocess_dag_for_batch(
            dag, dag_values, dag_grads
        )
        
        return construct_optimal_path_batch_fast(
            dag_values_array=dag_values_array,
            dag_grads_array=dag_grads_array,
            node_types=node_types,
            node_children=node_children,
            t_start=t_start,
            X_start=jnp.array(X_start),
            dag_root_id=dag_root_id,
            grid=grid,
            dynamics=dynamics,
            **kwargs
        )
    else:
        return construct_optimal_path_batch(
            dag=dag,
            dag_values=dag_values,
            dag_grads=dag_grads,
            t_start=t_start,
            X_start=X_start,
            dag_root_id=dag_root_id,
            grid=grid,
            dynamics=dynamics,
            **kwargs
        )
