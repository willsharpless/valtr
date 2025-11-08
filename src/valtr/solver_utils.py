import hj_reachability as hj
import hj_reachability.dynamics as dynamics
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm
from tqdm import tqdm

from valtr.reachability import DAGAvoid, DAGMaxN, DAGMinN, DAGNegate, DAGReachAvoid, DAGVar, dag_to_str, \
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

def solve_dag_values(dag_builder, dict_locals, grid_dict, dyn, times, gamma):
    
    dict_vars = {}
    number_of_solves = np.sum([1 for n in dag_builder.nodes if type(n) in [DAGReachAvoid, DAGAvoid]])
    print("Solving {} values at {} times...".format(number_of_solves, len(times)-1))
    with tqdm(total=number_of_solves * (len(times)-1)) as pbar:
        for dag_id, n in enumerate(dag_builder.nodes):
            match n:
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
                    fig_values = plot_values(temporal_values, times, grid_dict, title=title, dag_id=dag_id)

                case DAGAvoid(avoid=avoid):
                    # Note: the avoid is a stay since we are maximizing the value.
                    arg_avoid = dict_vars[avoid]

                    # Get a string representation of the sub-DAG for logging.
                    title = "%{}: {}".format(dag_id, dag_to_str(dag_builder, DAGId(dag_id)))

                    temporal_values = solve_avoid(arg_avoid, times=times, grid=grid_dict["grid"], dyn=dyn, gamma=gamma, pbar=pbar)
                    dict_vars[dag_id] = temporal_values[-1]  # Inf time approx final time
                    fig_values = plot_values(temporal_values, times, grid_dict, title=title, dag_id=dag_id)

    return dict_vars


def plot_values(values_over_time: list, times: np.ndarray, grid_params: dict, title: str, dag_id: int):
    figsize = (6 * len(times), 4)
    fig_, axes_ = plt.subplots(nrows=1, ncols=len(times), figsize=figsize)
    grid_slice = grid_params["grid_slice"]

    for ti, _ in enumerate(times):
        ax_ = axes_[ti]
        ax_.set_title(f"t = -{times[ti]:2.2f}")
        ax_.set_aspect("equal")
        ax_.set(xlim=(grid_params["lbs"][0] - grid_params["grid_pad"][0], grid_params["ubs"][0] + grid_params["grid_pad"][0]))

        im = ax_.contourf(grid_params["bb_X"], grid_params["bb_Y"], 
                          values_over_time[ti][grid_slice],
                          levels=25, cmap="RdBu", norm=CenteredNorm())
        ln = ax_.contour(grid_params["bb_X"], grid_params["bb_Y"], 
                         values_over_time[ti][grid_slice],
                         levels=0, colors="black", linewidths=2)
        cbar = fig_.colorbar(im, ax=ax_)
        cbar.add_lines(ln)

    fig_.suptitle(title)
    fig_.savefig("{:02}_values.pdf".format(dag_id), bbox_inches="tight")
    plt.close(fig_)

    return fig_

# 4d slice [:, :, int(grid_params["grid_nv"]/2), int(grid_params["grid_nq"]/2)]
# 3d slice [:, :, int(grid_params["grid_nq"]/2)]
# 2d slice [:, :]