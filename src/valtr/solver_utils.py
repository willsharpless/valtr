import hj_reachability as hj
import hj_reachability.dynamics as dynamics
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm
from tqdm import tqdm
import jax

from valtr.reachability import DAGAvoid, DAGMaxN, DAGMinN, DAGNegate, DAGReachAvoid, DAGReachAvoidLoop, DAGVar, DAGConst, dag_to_str, \
    lower_ir_to_dag, DAGId

def solve_avoid(bb_sdf_avoid: np.ndarray, times: np.ndarray, grid: hj.Grid, dyn: dynamics.Dynamics, gamma: float = 1, pbar=None):        
    def a_post_processor(t, v):
        return (1 - gamma) * bb_sdf_avoid + gamma * jnp.minimum(v, bb_sdf_avoid)

    solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=a_post_processor)
    values = init_values = bb_sdf_avoid

    values_over_time = [None] * len(times)
    for ti, _ in enumerate(times):
        if ti == 0:
            values = init_values
        else:
            values = hj.step(solver_settings, dyn, grid, -times[ti - 1], values, -times[ti], progress_bar=True)
            # clear last progress bar from terminal
            print("\033[2K\r", end="")  # Clear current line
            print("\033[1A\033[2K\r", end="")  # Move up one line and clear it
            if pbar is not None:
                pbar.update(1)

        values_over_time[ti] = values

    return values_over_time

def solve_reach_avoid(bb_sdf_reach: np.ndarray, bb_sdf_avoid: np.ndarray, times: np.ndarray, grid: hj.Grid, dyn: dynamics.Dynamics, gamma: float = 1, pbar=None):
    assert bb_sdf_reach.shape == bb_sdf_avoid.shape

    def ra_post_processor(t, v):
        return (1 - gamma) * jnp.minimum(bb_sdf_reach, bb_sdf_avoid) + gamma * jnp.minimum(jnp.maximum(v, bb_sdf_reach), bb_sdf_avoid)

    solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=ra_post_processor)
    values = init_values = bb_sdf_reach

    values_over_time = [None] * len(times)
    for ti, _ in enumerate(times):
        if ti == 0:
            values = init_values
        else:
            values = hj.step(solver_settings, dyn, grid, -times[ti - 1], values, -times[ti], progress_bar=True)
            # clear last progress bar from terminal
            print("\033[2K\r", end="")  # Clear current line
            print("\033[1A\033[2K\r", end="")  # Move up one line and clear it
            if pbar is not None:
                pbar.update(1)

        values_over_time[ti] = values

    return values_over_time

def solve_reach_avoid_loop(bb_sdf_reach: np.ndarray, bb_sdf_avoid: np.ndarray, times: np.ndarray, grid: hj.Grid, dyn: dynamics.Dynamics, gamma: float = 1, pbar=None):
    assert bb_sdf_reach.shape == bb_sdf_avoid.shape

    def ra_loop_post_processor(t, next_values, values):
        return (1 - gamma) * jnp.minimum(jnp.minimum(bb_sdf_reach, values), bb_sdf_avoid) + gamma * jnp.minimum(jnp.maximum(next_values, jnp.minimum(bb_sdf_reach, values)), bb_sdf_avoid)

    solver_settings = hj.SolverSettings.with_accuracy("very_high_passval", value_postprocessor=ra_loop_post_processor)
    values = init_values = bb_sdf_reach

    values_over_time = [None] * len(times)
    for ti, _ in enumerate(times):
        if ti == 0:
            values = init_values
        else:
            values = hj.step(solver_settings, dyn, grid, -times[ti - 1], values, -times[ti], progress_bar=True)
            # clear last progress bar from terminal
            print("\033[2K\r", end="")  # Clear current line
            print("\033[1A\033[2K\r", end="")  # Move up one line and clear it
            if pbar is not None:
                pbar.update(1)

        values_over_time[ti] = values
    # time here now corresponds to maximum time between recurrences

    return values_over_time

def solve_next_values(values: np.ndarray, next_time: float, grid: hj.Grid, dyn: dynamics.Dynamics, gamma: float = 1, tv:bool=False):

    # roll out one step, using dynamics optimal control
    init_x = grid.states
    grad_values = grid.grad_values(values)
        
    @jax.jit
    def dynamics_for_dag_id(states, times, grad_values):
        def single_dynamics(state, time, grad_values):
            if tv and times is not None:
                time_idx = jnp.argmin(jnp.abs(times - time))
                grad_value = grid.interpolate_fast_jit(grad_values[time_idx], state=state)
            else:
                grad_value = grid.interpolate_fast_jit(grad_values, state=state)
            
            u = dyn.optimal_control(state, time, grad_value)
            d = dyn.optimal_disturbance(state, time, grad_value)
            fx = dyn.open_loop_dynamics(state, time)
            Bu = dyn.control_jacobian(state, time)
            Bd = dyn.disturbance_jacobian(state, time)

            return fx + Bu @ u + Bd @ d

        return jax.vmap(single_dynamics)(states, times, grad_values)

    next_states = dynamics_for_dag_id(init_x, next_time, grad_values)
    next_values = grid.interpolate_fast_jit(values, state=next_states)

    return gamma * next_values

def solve_dag_values(dag_builder, dict_locals, grid_dict, dyn, times, gamma, 
                     multi_last_slice: bool = False, multi_slices: int = 5, multi_label: str = "Last Dim"):
    
    dict_vars = {}
    number_of_solves = np.sum([1 for n in dag_builder.nodes if type(n) in [DAGReachAvoid, DAGAvoid, DAGReachAvoidLoop]])
    print("Solving {} values at {} times...".format(number_of_solves, len(times)-1))
    with tqdm(total=number_of_solves * (len(times)-1)) as pbar:
        for dag_id, n in enumerate(dag_builder.nodes):
            match n:
                case DAGConst(value=value):
                    first_local_var = next(iter(dict_locals.values()))
                    dict_vars[dag_id] = np.inf * np.ones_like(first_local_var) # dummy array with correct shape

                case DAGVar(name=name):
                    assert name in dict_locals, "Unknown variable name {}".format(name)
                    dict_vars[dag_id] = dict_locals[name]

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

                    # Get a string representation of the sub-DAG for logging.
                    title = "%{}: {}".format(dag_id, dag_to_str(dag_builder, DAGId(dag_id)))

                    temporal_values = solve_reach_avoid(arg_reach, arg_avoid, times=times, grid=grid_dict["grid"], dyn=dyn, gamma=gamma, pbar=pbar)
                    dict_vars[dag_id] = temporal_values[-1]  # Inf time approx final time

                case DAGReachAvoidLoop(reach=reach, avoid=avoid):
                    # Note: the avoid is a stay since we are maximizing the value.
                    arg_reach = dict_vars[reach]
                    arg_avoid = dict_vars[avoid]

                    # Get a string representation of the sub-DAG for logging.
                    title = "%{}: {}".format(dag_id, dag_to_str(dag_builder, DAGId(dag_id)))

                    temporal_values = solve_reach_avoid_loop(arg_reach, arg_avoid, times=times, grid=grid_dict["grid"], dyn=dyn, gamma=gamma, pbar=pbar)
                    dict_vars[dag_id] = temporal_values[-1]  # Inf time approx final time

                case DAGAvoid(avoid=avoid):
                    # Note: the avoid is a stay since we are maximizing the value.
                    arg_avoid = dict_vars[avoid]

                    # Get a string representation of the sub-DAG for logging.
                    title = "%{}: {}".format(dag_id, dag_to_str(dag_builder, DAGId(dag_id)))

                    temporal_values = solve_avoid(arg_avoid, times=times, grid=grid_dict["grid"], dyn=dyn, gamma=gamma, pbar=pbar)
                    dict_vars[dag_id] = temporal_values[-1]  # Inf time approx final time
            
            if type(n) in [DAGReachAvoid, DAGAvoid, DAGReachAvoidLoop]:
                fig_values = plot_values(temporal_values, times, grid_dict, title=title, dag_id=dag_id, 
                                          multi_last_slice=multi_last_slice, multi_slices=multi_slices, multi_label=multi_label)

    return dict_vars


def plot_values(values_over_time: list, times: np.ndarray, grid_params: dict, title: str, dag_id: int, 
                multi_last_slice: bool = False, multi_slices: int = 5, multi_label: str = "Last Dim") -> plt.Figure:

    grid_slice = grid_params["grid_slice"]

    if not multi_last_slice:
        figsize = (6 * len(times), 4)
        fig_, axes_ = plt.subplots(nrows=1, ncols=len(times), figsize=figsize, constrained_layout=True)
        for ti, _ in enumerate(times):
            ax_ = axes_[ti]
            ax_.set_title(f"t = -{times[ti]:2.2f}")
            ax_.set_aspect("equal")
            ax_.set(xlim=(grid_params["lbs"][0] - grid_params["grid_pad"][0], grid_params["ubs"][0] + grid_params["grid_pad"][0]))

            im = ax_.contourf(grid_params["grid_X"], grid_params["grid_Y"], 
                            values_over_time[ti][grid_slice],
                            levels=25, cmap="RdBu", norm=CenteredNorm())
            ln = ax_.contour(grid_params["grid_X"], grid_params["grid_Y"], 
                            values_over_time[ti][grid_slice],
                            levels=0, colors="black", linewidths=2)
            cbar = fig_.colorbar(im, ax=ax_)
            cbar.add_lines(ln)

    else:
        figsize = (6 * len(times), 4 * multi_slices)
        fig_, axes_ = plt.subplots(nrows=multi_slices, ncols=len(times), figsize=figsize, constrained_layout=True)
        for ti, _ in enumerate(times):
            for si in range(multi_slices):
                ax_ = axes_[si, ti]
                ax_.set_aspect("equal")
                ax_.set(xlim=(grid_params["lbs"][0] - grid_params["grid_pad"][0], grid_params["ubs"][0] + grid_params["grid_pad"][0]))

                last_dim_len = grid_params["grid"].shape[-1] - 1
                slice_index = int(last_dim_len / multi_slices * (si + 0.5))
                grid_slice_multi = tuple(list(grid_slice[:-1]) + [slice_index])
                im = ax_.contourf(grid_params["grid_X"], grid_params["grid_Y"], 
                                values_over_time[ti][grid_slice_multi],
                                levels=25, cmap="RdBu", norm=CenteredNorm())
                ln = ax_.contour(grid_params["grid_X"], grid_params["grid_Y"], 
                                values_over_time[ti][grid_slice_multi],
                                levels=0, colors="black", linewidths=2)
                ax_.set_title(f"t = -{times[ti]:2.2f}, {multi_label} = {grid_params['grid'].coordinate_vectors[-1][slice_index]:.2f}")
                
                cbar = fig_.colorbar(im, ax=ax_)
                cbar.add_lines(ln)

    fig_.suptitle(title)
    fig_.savefig("node_values/node{:02}_values.pdf".format(dag_id), bbox_inches="tight")
    plt.close(fig_)

    return fig_

# 4d slice [:, :, int(grid_params["grid_nv"]/2), int(grid_params["grid_nq"]/2)]
# 3d slice [:, :, int(grid_params["grid_nq"]/2)]
# 2d slice [:, :]