import hj_reachability as hj
import hj_reachability.dynamics as dynamics
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm
from tqdm import tqdm
import jax

from valtr.reachability import DAGAvoid, DAGMaxN, DAGMinN, DAGNegate, DAGNext, DAGReachAvoid, DAGReachAvoidLoop, DAGVar, DAGConst, dag_to_str, \
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

def solve_next_values(values: np.ndarray, time: float, grid: hj.Grid, dyn: dynamics.Dynamics, 
                      step_size: float = 0.1, gamma: float = 1., tv: bool = False, smooth: bool = True):
    if tv:
        raise NotImplementedError("Time varying dynamics not implemented yet.")
    
    grad_values = grid.grad_values(values)
    
    def step(state):
        grad_value = grid.interpolate_fast_jit(grad_values, state=state)
        
        u = dyn.optimal_control(state, time, grad_value)
        d = dyn.optimal_disturbance(state, time, grad_value)
        fx = dyn.open_loop_dynamics(state, time)
        Bu = dyn.control_jacobian(state, time)
        Bd = dyn.disturbance_jacobian(state, time)

        dx = fx + Bu @ u + Bd @ d

        # next_state = state + step_size * dx # Forward Euler
        # rk45
        k1 = dx
        k2 = dyn.open_loop_dynamics(state + 0.5 * step_size * k1, time + 0.5 * step_size)
        k2 += dyn.control_jacobian(state + 0.5 * step_size * k1, time + 0.5 * step_size) @ u
        k2 += dyn.disturbance_jacobian(state + 0.5 * step_size * k1, time + 0.5 * step_size) @ d
        k3 = dyn.open_loop_dynamics(state + 0.5 * step_size * k2, time + 0.5 * step_size)
        k3 += dyn.control_jacobian(state + 0.5 * step_size * k2, time + 0.5 * step_size) @ u
        k3 += dyn.disturbance_jacobian(state + 0.5 * step_size * k2, time + 0.5 * step_size) @ d
        k4 = dyn.open_loop_dynamics(state + step_size * k3, time + step_size)
        k4 += dyn.control_jacobian(state + step_size * k3, time + step_size) @ u
        k4 += dyn.disturbance_jacobian(state + step_size * k3, time + step_size) @ d
        next_state = state + (step_size / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

        return next_state

    next_states = jax.vmap(jax.vmap(step))(grid.states)

    # Clip next_states to be within grid bounds ("G in_grid" makes this ok)
    next_states_clipped = jnp.clip(next_states, grid.domain.lo, grid.domain.hi)
    next_values = jax.vmap(jax.vmap(lambda state: grid.interpolate_fast_jit(values, state=state)))(next_states_clipped)

    # perform a convolution to smooth out any artifacts from interpolation
    # if smooth:
    #     # kernel = jnp.array([[0, 1, 0],
    #     #                     [1, 4, 1],
    #     #                     [0, 1, 0]]) / 8.0
    #     kernel = jnp.array([[1, 2, 3, 2, 1],
    #                         [2, 4, 6, 4, 2],
    #                         [3, 6, 9, 6, 3],
    #                         [2, 4, 6, 4, 2],
    #                         [1, 2, 3, 2, 1]]) / 16.0
    #     next_values = jax.scipy.signal.convolve2d(next_values, kernel, mode='same')

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

                case DAGNext(arg=arg):
                    val = dict_vars[arg]
                    next_time = times[-1] # FIXME for tv systems
                    next_val = solve_next_values(val, next_time, grid_dict["grid"], dyn, gamma=gamma)
                    dict_vars[dag_id] = next_val

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

                    temporal_values = solve_reach_avoid(arg_reach, arg_avoid, times=times, grid=grid_dict["grid"], dyn=dyn, gamma=gamma, pbar=pbar)
                    dict_vars[dag_id] = temporal_values[-1]  # Inf time approx final time

                case DAGReachAvoidLoop(reach=reach, avoid=avoid):
                    # Note: the avoid is a stay since we are maximizing the value.
                    arg_reach = dict_vars[reach]
                    arg_avoid = dict_vars[avoid]

                    temporal_values = solve_reach_avoid_loop(arg_reach, arg_avoid, times=times, grid=grid_dict["grid"], dyn=dyn, gamma=gamma, pbar=pbar)
                    dict_vars[dag_id] = temporal_values[-1]  # Inf time approx final time

                case DAGAvoid(avoid=avoid):
                    # Note: the avoid is a stay since we are maximizing the value.
                    arg_avoid = dict_vars[avoid]

                    temporal_values = solve_avoid(arg_avoid, times=times, grid=grid_dict["grid"], dyn=dyn, gamma=gamma, pbar=pbar)
                    dict_vars[dag_id] = temporal_values[-1]  # Inf time approx final time
            
            title = "%{}: {}".format(dag_id, dag_to_str(dag_builder, DAGId(dag_id)))
            if type(n) in [DAGReachAvoid, DAGAvoid, DAGReachAvoidLoop]:
                fig_values = plot_values(temporal_values, times, grid_dict, title=title, dag_id=dag_id, 
                                          multi_last_slice=multi_last_slice, multi_slices=multi_slices, multi_label=multi_label)
            if type(n) in [DAGNext]:
                fig_values = plot_values_next(val, next_val, times[-1], grid_dict, title=title, dag_id=dag_id)

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

def plot_values_next(values: np.ndarray, next_values: np.ndarray, final_time:float, grid_params: dict, title: str, dag_id: int) -> plt.Figure:

    grid_slice = grid_params["grid_slice"]

    figsize = (6*3, 4)
    fig_, axes_ = plt.subplots(nrows=1, ncols=3, figsize=figsize, constrained_layout=True)
    for vi, value in enumerate([values, next_values, next_values - values]):
        ax_ = axes_[vi]
        ax_.set_title(f"t = -{final_time:2.2f}, {['Before', 'Next', 'Difference'][vi]}")
        ax_.set_aspect("equal")
        ax_.set(xlim=(grid_params["lbs"][0] - grid_params["grid_pad"][0], grid_params["ubs"][0] + grid_params["grid_pad"][0]))

        im = ax_.contourf(grid_params["grid_X"], grid_params["grid_Y"], 
                        value[grid_slice],
                        levels=100, cmap="RdBu" if vi < 2 else "Spectral",
                        norm=CenteredNorm())
        ln = ax_.contour(grid_params["grid_X"], grid_params["grid_Y"], 
                        value[grid_slice],
                        levels=0, colors="black", linewidths=2)
            
        cbar = fig_.colorbar(im, ax=ax_)
        cbar.add_lines(ln)

    fig_.suptitle(title)
    fig_.savefig("node_values/node{:02}_values.pdf".format(dag_id), bbox_inches="tight")
    plt.close(fig_)

    return fig_

# 4d slice [:, :, int(grid_params["grid_nv"]/2), int(grid_params["grid_nq"]/2)]
# 3d slice [:, :, int(grid_params["grid_nq"]/2)]
# 2d slice [:, :]