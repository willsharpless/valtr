import copy
import os

import hj_reachability as hj
import hj_reachability.dynamics as dynamics
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import CenteredNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate as ode

from valtr.control import construct_optimal_path, plot_optimal_path
from valtr.dag_graphviz import visualize_dag
from valtr.dag_passes import PassFoldConstBool
from valtr.ir_builder import IRBuilder
from valtr.ir_pass import PassCombineGloballySegments, PassFinallyToUntil
from valtr.lowering import Lowerer
from valtr.reachability import (DAGAvoid, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGReachAvoid, DAGVar, dag_to_str,
                                lower_ir_to_dag)
from valtr.solver_utils import solve_dag_values
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser
from valtr.util.jax_util import rep_vmap


class Drift(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, u_bd=0.0, d_bd=0.0, N=1, control_mode="max", disturbance_mode="min"):
        self.N = N
        self.dim = 2 * N
        control_space = hj.sets.Ball(jnp.array([0, 0]), u_bd)
        disturbance_space = hj.sets.Ball(jnp.array([0, 0]), d_bd)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        assert state.shape == (2,)
        return jnp.sign(state)

    def control_jacobian(self, state, time):
        return 0 * jnp.eye(self.dim)

    def disturbance_jacobian(self, state, time):
        return jnp.eye(self.dim)


def main():
    lbs = np.array([-2.0, -2.0])
    ubs = np.array([2.0, 2.0])
    grid_pad = np.array([0.2, 0.2])
    grid_nx = 501
    grid_ny = 501
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lbs - grid_pad, ubs + grid_pad), [grid_nx, grid_ny]
    )

    gamma = 0.9999999

    def sdf_r(pt: np.ndarray):
        # return pt[1] - 1
        return jnp.where(pt[1] >= 1, 0.1, -10.0)

    def sdf_q(pt: np.ndarray):
        return -(pt[0] - 1)

    def jit_rep_vmap(fn, rep: int, *args):
        return np.array(rep_vmap(fn, rep=rep)(*args))

    bb_pos = np.array(grid.states)
    bb_sdf_r = jit_rep_vmap(sdf_r, 2, bb_pos)
    bb_sdf_q = jit_rep_vmap(sdf_q, 2, bb_pos)

    bb_x = bb_pos[:, :, 0]
    bb_y = bb_pos[:, :, 1]

    fig, axes = plt.subplots(1, 2, layout="constrained")
    ax = axes[0]
    ax.set_aspect("equal")
    cmap = "RdBu"
    norm = CenteredNorm()
    im = ax.contourf(bb_x, bb_y, bb_sdf_r, levels=25, norm=norm, cmap=cmap)
    ln = ax.contour(bb_x, bb_y, bb_sdf_r, levels=0, colors="black", linewidths=2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.add_lines(ln)
    ax.set_title("sdf_r")

    ax = axes[1]
    ax.set_aspect("equal")
    norm = CenteredNorm()
    im = ax.contourf(bb_x, bb_y, bb_sdf_q, levels=25, norm=norm, cmap=cmap)
    ln = ax.contour(bb_x, bb_y, bb_sdf_q, levels=0, colors="black", linewidths=2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.add_lines(ln)
    ax.set_title("sdf_q")

    fig_path = "sdfs.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    # ----

    # def ra_post_processor(t, v):
    #     return (1 - gamma) * jnp.maximum(bb_sdf_q, bb_sdf_r) + gamma * jnp.maximum(jnp.minimum(v, bb_sdf_q), bb_sdf_r)

    def ra_post_processor(t, v):
        return jnp.maximum(jnp.minimum(v, bb_sdf_q), bb_sdf_r)

    solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=ra_post_processor)
    values = init_values = bb_sdf_q

    dyn = Drift()
    ntimes = 4
    times = np.linspace(0.0, 5.0, ntimes)

    values_over_time = [None] * len(times)
    for ti, _ in enumerate(times):
        if ti == 0:
            values = init_values
        else:
            values = hj.step(solver_settings, dyn, grid, -times[ti - 1], values, -times[ti], progress_bar=True)

        values_over_time[ti] = values

    figsize = (6 * len(times), 4)
    fig_, axes_ = plt.subplots(nrows=1, ncols=len(times), figsize=figsize)
    for ti, _ in enumerate(times):
        ax_ = axes_[ti]
        ax_.set_title(f"t = -{times[ti]:2.2f}")
        ax_.set_aspect("equal")
        norm = CenteredNorm()
        im = ax_.contourf(bb_x, bb_y, values_over_time[ti], levels=25, norm=norm, cmap=cmap)
        ln = ax_.contour(bb_x, bb_y, values_over_time[ti], levels=0, colors="black", linewidths=2)
        cbar = fig_.colorbar(im, ax=ax_)
        cbar.add_lines(ln)

    fig_.savefig("release.pdf", bbox_inches="tight")
    plt.close(fig_)


if __name__ == "__main__":
    main()
