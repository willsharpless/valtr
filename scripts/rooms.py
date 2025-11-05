import hj_reachability as hj
import hj_reachability.dynamics as dynamics
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import CenteredNorm, ListedColormap

from valtr.dag_graphviz import visualize_dag
from valtr.dag_passes import PassFoldConstBool
from valtr.ir_builder import IRBuilder
from valtr.ir_pass import PassCombineGloballySegments, PassFinallyToUntil
from valtr.lowering import Lowerer
from valtr.reachability import DAGAvoid, DAGMaxN, DAGMinN, DAGNegate, DAGReachAvoid, DAGVar, dag_to_str, \
    lower_ir_to_dag, DAGId
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser
from valtr.util.jax_util import rep_vmap
from scipy import integrate as ode


class Point(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, u_bd=1.0, d_bd=0.0, N=1, control_mode="max", disturbance_mode="min"):
        self.N = N
        self.dim = 2 * N
        # control_space = hj.sets.Box(-u_bd * jnp.ones(2 * N), u_bd * jnp.ones(2 * N))
        control_space = hj.sets.Ball(jnp.array([0, 0]), u_bd)
        # disturbance_space = hj.sets.Box(-d_bd * jnp.ones(2 * N), d_bd * jnp.ones(2 * N))
        disturbance_space = hj.sets.Ball(jnp.array([0, 0]), d_bd)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.zeros_like(state)

    def control_jacobian(self, state, time):
        return jnp.eye(self.dim)

    def disturbance_jacobian(self, state, time):
        return jnp.eye(self.dim)


def sdf_aabb(pt, center, halfw_x, halfw_y):
    dx = jnp.abs(pt[0] - center[0]) - halfw_x
    dy = jnp.abs(pt[1] - center[1]) - halfw_y
    dx_clamped = jnp.maximum(dx, 0.0)
    dy_clamped = jnp.maximum(dy, 0.0)
    outside_dist = jnp.sqrt(dx_clamped**2 + dy_clamped**2)
    inside_dist = jnp.maximum(dx, dy)
    return outside_dist + inside_dist


def sdf_aabb_blcorner(pt, bl_corner, halfw_x, halfw_y):
    center = bl_corner + jnp.array([halfw_x, halfw_y])
    return sdf_aabb(pt, center, halfw_x, halfw_y)


def sdf_circle(pt, center, radius):
    return jnp.linalg.norm(pt - center) - radius


def main():
    lbs = np.array([-1.0, 0.0])
    ubs = np.array([2.0, 1.0])
    grid_pad = np.array([0.2, 0.2])

    grid_nx = 501
    grid_ny = 201

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lbs - grid_pad, ubs + grid_pad), [grid_nx, grid_ny]
    )

    # (grid_nx, grid_ny)
    bb_pos = np.array(grid.states)

    # (grid_nx, )
    b_x = np.array(grid.coordinate_vectors[0])
    b_y = np.array(grid.coordinate_vectors[1])

    bb_X = bb_pos[:, :, 0]
    bb_Y = bb_pos[:, :, 1]

    # ----------------------------------------------------------------
    # SDF for room 1: BL corner at (0, 0), width = height = 1
    def sdf_room1(pt):
        return sdf_aabb_blcorner(pt, jnp.array([0.0, 0.0]), 0.5, 0.5)

    # Room2 is to the right of room1. Also width = height = 1.
    def sdf_room2(pt):
        return sdf_aabb_blcorner(pt, jnp.array([1.0, 0.0]), 0.5, 0.5)

    # Room3 is to the left of room1. Also width = height = 1.
    def sdf_room3(pt):
        return sdf_aabb_blcorner(pt, jnp.array([-1.0, 0.0]), 0.5, 0.5)

    # door12 is between room1 and room2. width=0.1, height of 1.0. Centered between the two rooms.
    def sdf_door12(pt):
        halfw_x = 0.05
        return sdf_aabb_blcorner(pt, jnp.array([0.95, 0.0]), halfw_x, 0.5)

    # door13 is between room1 and room3. width=0.1, height of 1.0. Centered between the two rooms.
    def sdf_door13(pt):
        halfw_x = 0.05
        return sdf_aabb_blcorner(pt, jnp.array([0.0 - halfw_x, 0.0]), halfw_x, 0.5)

    wall_thickness = 0.1

    # wallB is below all three rooms. width=3.0, height=wall_thickness. Centered below the three rooms.
    def sdf_wallB(pt):
        return sdf_aabb_blcorner(pt, jnp.array([-1.0, -wall_thickness]), 1.5, wall_thickness / 2)

    # sdf_wallT is above all three rooms. width=3.0, height=wall_thickness. Centered above the three rooms.
    def sdf_wallT(pt):
        return sdf_aabb_blcorner(pt, jnp.array([-1.0, 1.0]), 1.5, wall_thickness / 2)

    # sdf_wallL is to the left of all three rooms. width=wall_thickness, height=1.0. Centered to the left of the three rooms.
    def sdf_wallL(pt):
        return sdf_aabb_blcorner(pt, jnp.array([-1.0 - wall_thickness, 0.0]), wall_thickness / 2, 0.5)

    def sdf_wallR(pt):
        return sdf_aabb_blcorner(pt, jnp.array([2.0, 0.0]), wall_thickness / 2, 0.5)

    # Key1 is in room1, Key2 is in room 2, Key3 is in room3.
    def sdf_key1(pt):
        return sdf_circle(pt, jnp.array([0.2, 0.6]), 0.05)

    def sdf_key2(pt):
        return sdf_circle(pt, jnp.array([1.75, 0.2]), 0.05)

    def sdf_key3(pt):
        return sdf_circle(pt, jnp.array([-0.6, 0.7]), 0.05)

    def jit_rep_vmap(fn, rep: int, *args):
        return np.array(rep_vmap(fn, rep=rep)(*args))

    # ------------------------------------------------------------------
    bb_sdf_room1 = -jit_rep_vmap(sdf_room1, 2, bb_pos)
    bb_sdf_room2 = -jit_rep_vmap(sdf_room2, 2, bb_pos)
    bb_sdf_room3 = -jit_rep_vmap(sdf_room3, 2, bb_pos)

    bb_sdf_door12 = -jit_rep_vmap(sdf_door12, 2, bb_pos)
    bb_sdf_door13 = -jit_rep_vmap(sdf_door13, 2, bb_pos)

    bb_sdf_wallB = -jit_rep_vmap(sdf_wallB, 2, bb_pos)
    bb_sdf_wallT = -jit_rep_vmap(sdf_wallT, 2, bb_pos)
    bb_sdf_wallL = -jit_rep_vmap(sdf_wallL, 2, bb_pos)
    bb_sdf_wallR = -jit_rep_vmap(sdf_wallR, 2, bb_pos)

    bb_sdf_key1 = -jit_rep_vmap(sdf_key1, 2, bb_pos)
    bb_sdf_key2 = -jit_rep_vmap(sdf_key2, 2, bb_pos)
    bb_sdf_key3 = -jit_rep_vmap(sdf_key3, 2, bb_pos)

    # bb_sdf_room1 = rep_vmap(sdf_room1, rep=2)(bb_pos)
    # bb_sdf_room2 = rep_vmap(sdf_room2, rep=2)(bb_pos)
    # bb_sdf_room3 = rep_vmap(sdf_room3, rep=2)(bb_pos)
    #
    # bb_sdf_door12 = rep_vmap(sdf_door12, rep=2)(bb_pos)
    # bb_sdf_door13 = rep_vmap(sdf_door13, rep=2)(bb_pos)
    #
    # bb_sdf_wallB = rep_vmap(sdf_wallB, rep=2)(bb_pos)
    # bb_sdf_wallT = rep_vmap(sdf_wallT, rep=2)(bb_pos)
    # bb_sdf_wallL = rep_vmap(sdf_wallL, rep=2)(bb_pos)
    # bb_sdf_wallR = rep_vmap(sdf_wallR, rep=2)(bb_pos)

    # ------------------------------------------------------------------
    def shade_supzero(ax_: plt.Axes, bb_sdf, color, alpha: float = 0.5):
        # Mask negative values
        masked = np.ma.array(bb_sdf, mask=bb_sdf < 0)

        # Compute extent from coordinate grid
        xmin, xmax = bb_X.min(), bb_X.max()
        ymin, ymax = bb_Y.min(), bb_Y.max()

        ax_.imshow(
            masked.T,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            cmap=ListedColormap([color]),
            alpha=alpha,
            interpolation="nearest",
        )

    # def shade_supzero(ax_: plt.Axes, bb_sdf, color, alpha: float = 0.5):
    #     ipdb.set_trace()
    #     ax_.pcolormesh(
    #         bb_X,
    #         bb_Y,
    #         np.ma.array(bb_sdf, mask=bb_sdf < 0),
    #         shading="auto",
    #         cmap=ListedColormap([color]),
    #         alpha=alpha,
    #     )

    logger.info("Plotting SDFs...")
    fig_rooms, ax_rooms = plt.subplots(layout="constrained")
    ax_rooms.set_aspect("equal")

    # Shade inside the rooms.
    shade_supzero(ax_rooms, bb_sdf_room1, "C1", alpha=0.5)
    shade_supzero(ax_rooms, bb_sdf_room2, "C2", alpha=0.5)
    shade_supzero(ax_rooms, bb_sdf_room3, "C4", alpha=0.5)

    shade_supzero(ax_rooms, bb_sdf_door12, "C5", alpha=0.8)
    shade_supzero(ax_rooms, bb_sdf_door13, "C5", alpha=0.8)

    shade_supzero(ax_rooms, bb_sdf_wallB, "k", alpha=0.8)
    shade_supzero(ax_rooms, bb_sdf_wallT, "k", alpha=0.8)
    shade_supzero(ax_rooms, bb_sdf_wallL, "k", alpha=0.8)
    shade_supzero(ax_rooms, bb_sdf_wallR, "k", alpha=0.8)

    shade_supzero(ax_rooms, bb_sdf_key1, "C6", alpha=0.8)
    shade_supzero(ax_rooms, bb_sdf_key2, "C6", alpha=0.8)
    shade_supzero(ax_rooms, bb_sdf_key3, "C6", alpha=0.8)

    fig_path = "rooms_sdf.pdf"
    fig_rooms.savefig(fig_path, bbox_inches="tight")
    # plt.close(fig_rooms)
    # -------------------------------------------------------------------------------------------
    # Build solve_reach_avoid and solve_avoid functions.
    dyn = Point()

    tf = 2.0
    ntimes = 5
    times = np.linspace(0.0, tf, ntimes)
    gamma = 0.999

    def solve_avoid(bb_sdf_avoid: np.ndarray, title: str, dag_id: int):
        def post_processor(t, v):
            assert v.shape == bb_sdf_avoid.shape
            # return jnp.minimum(v, bb_sdf_avoid)
            return jnp.exp((1 - gamma) * t) * jnp.minimum(v, bb_sdf_avoid)

        solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=post_processor)

        values = init_values = bb_sdf_avoid

        figsize = (6 * ntimes, 4)
        fig_, axes_ = plt.subplots(nrows=1, ncols=ntimes, figsize=figsize)

        for ti, _ in enumerate(times):
            ax_ = axes_[ti]
            ax_.set_title(f"t = -{times[ti]:2.2f}")
            ax_.set_aspect("equal")
            ax_.set(xlim=(lbs[0] - grid_pad[0], ubs[0] + grid_pad[0]))

            if ti == 0:
                values = init_values
            else:
                values = hj.step(solver_settings, dyn, grid, -times[ti - 1], values, -times[ti], progress_bar=True)

            cmap = "RdBu"
            norm = CenteredNorm()
            im = ax_.contourf(bb_X, bb_Y, values, levels=25, cmap=cmap, norm=norm)
            ln = ax_.contour(bb_X, bb_Y, values, levels=0, colors="black", linewidths=2)
            cbar = fig_.colorbar(im, ax=ax_)
            cbar.add_lines(ln)

        fig_.suptitle(title)
        fig_.savefig("{:02}_avoid.pdf".format(dag_id), bbox_inches="tight")
        plt.close(fig_)

        return values

    def solve_reach_avoid(bb_sdf_reach: np.ndarray, bb_sdf_avoid: np.ndarray, title: str, dag_id: int):
        def post_processor(t, v):
            assert v.shape == bb_sdf_reach.shape == bb_sdf_avoid.shape
            # return jnp.minimum(jnp.maximum(v, bb_sdf_reach), bb_sdf_avoid)
            return jnp.exp((1 - gamma) * t) * jnp.minimum(jnp.maximum(v, bb_sdf_reach), bb_sdf_avoid)

        solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=post_processor)
        values = init_values = bb_sdf_reach

        figsize = (6 * ntimes, 4)
        fig_, axes_ = plt.subplots(nrows=1, ncols=ntimes, figsize=figsize)

        for ti, _ in enumerate(times):
            ax_ = axes_[ti]
            ax_.set_title(f"t = -{times[ti]:2.2f}")
            ax_.set_aspect("equal")
            ax_.set(xlim=(lbs[0] - grid_pad[0], ubs[0] + grid_pad[0]))

            if ti == 0:
                values = init_values
            else:
                values = hj.step(solver_settings, dyn, grid, -times[ti - 1], values, -times[ti], progress_bar=True)

            cmap = "RdBu"
            norm = CenteredNorm()
            im = ax_.contourf(bb_X, bb_Y, values, levels=25, cmap=cmap, norm=norm)
            ln = ax_.contour(bb_X, bb_Y, values, levels=0, colors="black", linewidths=2)
            cbar = fig_.colorbar(im, ax=ax_)
            cbar.add_lines(ln)

        fig_.suptitle(title)
        fig_.savefig("{:02}_reach_avoid.pdf".format(dag_id), bbox_inches="tight")
        plt.close(fig_)

        return values

    # --------------------------------------------------------------------------------
    # source = "(!d1 U k1) && (!d2 U k2) && F k3 && G( !wB ) && G( !wT ) && G( !wL ) && G( !wR )"
    source = "(!d1 U k1) && (!d2 U k2) && G( !wB ) && G( !wT ) && G( !wL ) && G( !wR )"
    # source = "(!d1 U k1) && G( !wB ) && G( !wT ) && G( !wL ) && G( !wR )"

    lexer = TLLexer()
    tokens = list(lexer.tokenize(source))
    ast = TLParser(tokens).parse()

    # AST -> IR
    ir_builder = IRBuilder()
    lowerer = Lowerer(builder=ir_builder)
    ir_root_id = lowerer.lower(ast)

    passes = [PassFinallyToUntil, PassCombineGloballySegments]
    for p_cls in passes:
        p = p_cls(ir_builder)
        ir_root_id, ir_builder = p.run(ir_root_id)

    # IR -> DAG
    dag_builder, dag_root = lower_ir_to_dag(ir_builder, ir_root_id)

    # Perform constant folding.
    passes = [PassFoldConstBool]
    for p_cls in passes:
        p = p_cls(dag_builder)
        dag_root, dag_builder, changed = p.run(dag_root)

    # Visualize the DAG.
    dot_dag = visualize_dag(dag_builder, dag_root, filename="dag_graph", view=True)

    # -------------------------------------------------------------------------------------------
    # Interpret the DAG.

    #     Input variables into the expression.
    #     Positive sdf is truthy.
    dict_locals = {
        "k1": bb_sdf_key1,
        "d1": bb_sdf_door12,
        "k2": bb_sdf_key2,
        "d2": bb_sdf_door13,
        # ----
        "wB": bb_sdf_wallB,
        "wT": bb_sdf_wallT,
        "wL": bb_sdf_wallL,
        "wR": bb_sdf_wallR,
    }

    # Iterate through the DAG.
    dict_vars = {}

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

                val = solve_reach_avoid(arg_reach, arg_avoid, title=title, dag_id=dag_id)
                dict_vars[dag_id] = val

            case DAGAvoid(avoid=avoid):
                # Note: the avoid is a stay since we are maximizing the value.
                arg_avoid = dict_vars[avoid]

                # Get a string representation of the sub-DAG for logging.
                title = "%{}: {}".format(dag_id, dag_to_str(dag_builder, DAGId(dag_id)))

                val = solve_avoid(arg_avoid, title=title, dag_id=dag_id)
                dict_vars[dag_id] = val

    # Final result is at dag_root
    bb_sdf_result = dict_vars[dag_root]

    # Plot the final result.
    fig, ax = plt.subplots(layout="constrained")
    cmap = "RdBu"
    norm = CenteredNorm()
    im = ax.contourf(bb_X, bb_Y, bb_sdf_result, levels=25, cmap=cmap, norm=norm)
    ln = ax.contour(bb_X, bb_Y, bb_sdf_result, levels=0, colors="black", linewidths=2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.add_lines(ln)

    fig.suptitle("{}".format(source))
    fig.savefig("sol.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------------------------
    # Optimal paths from solutions

    # Make dict of value tree gradients, assuming time-invariant for now
    value_tree_grads = {}
    for key, val in dict_vars.items():
        value_tree_grads[key] = grid.grad_values(val)
        # value_tree_grads[key] = [grid.grad_values(val[i,...]) for i in range(len(times))]

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
        Bu = dynamics.control_jacobian(x,t)
        Bd = dynamics.disturbance_jacobian(x,t)
        
        dx = fx + Bu @ u + Bd @ d
        dx = dx.tolist()
        return dx

    def characteristic(t0, x0, grad_values, grid, dynamics, times=None, tv=False):
        sol = ode.solve_ivp(lambda t,x : model(t, x, grad_values, grid, dynamics, times=times, tv=tv), [t0, 0], x0, max_step = .1)
        return sol

    def construct_optimal_path(dag, dag_values, dag_grads, t_start, x_start, dag_id, tv=False, reaching_eps=0.):

        print("Rolling out - NODE [{}:{}] at t = {:2.2f}, x = {}".format(
            str(type(dag.nodes[dag_id])).split('.')[-1].split("'")[0],
            dag_id, t_start, x_start))
        dag_path, switch_times = [dag_id], []
        grad_values = dag_grads[dag_id]
        sol = characteristic(t_start, x_start, grad_values, grid, dyn, times=times, tv=tv)

        # Recursively overwrite sol until at an Avoid node (terminal)
        if type(dag.nodes[dag_id]) is DAGAvoid: #or type(dag_builder.nodes[dag_id]) is DAGReach:
            dag_path = [dag_id]
            switch_times = []
        
        # For each ReachAvoid node, find which child gives the earliest reaching value
        else:
            if type(dag.nodes[dag.nodes[dag_id].reach]) is DAGMaxN:
                next_dag_ids = dag.nodes[dag.nodes[dag_id].reach].args
            elif type(dag.nodes[dag.nodes[dag_id].reach]) is DAGMinN:
                next_dag_ids = [dag.nodes[dag_id].reach]
            else:
                raise RuntimeError("Expected Min/Max node in DAGReachAvoid.reach but got: [{}:{}]".format(
                    str(type(dag.nodes[dag_id])).split('.')[-1].split("'")[0], dag_id))
            best_next_dag_id = None
            best_reach_index = np.inf

            for next_dag_id in next_dag_ids:
                values_next = dag_values[next_dag_id]
                sol_values = np.array([grid.interpolate(values_next, state = sol.y[:,i]) for i in range(len(sol.t))])
                reach_index = np.argmax(sol_values > reaching_eps) if any(sol_values > reaching_eps) else np.inf

                if reach_index < best_reach_index:
                    best_reach_index = reach_index
                    best_next_dag_id = next_dag_id

            # If reaching child, recurse
            if best_next_dag_id is not None:
                dag_path.append(dag.nodes[dag_id].reach)

                # Pick next RA/A problem #FIXME? more complex for (UNTIL) UNTIL (UNTIL) etc...
                next_node = dag.nodes[best_next_dag_id]
                if not (type(next_node) is DAGAvoid or type(next_node) is DAGReachAvoid):
                    dag_path.append(best_next_dag_id) # will be overwritten
                    children_types = [type(dag.nodes[child]) for child in next_node.args]
                    if DAGReachAvoid in children_types:
                        best_next_dag_id = next_node.args[children_types.index(DAGReachAvoid)]
                    elif DAGAvoid in children_types:
                        best_next_dag_id = next_node.args[children_types.index(DAGAvoid)]
                    else:
                        raise RuntimeError("Could not find next ReachAvoid or Avoid node in children of node [{}:{}]".format(
                            str(type(next_node)).split('.')[-1].split("'")[0], best_next_dag_id))

                t_reach = sol.t[best_reach_index]
                x_reach = sol.y[:, best_reach_index]
                next_sol, next_dag_path, next_switch_times = construct_optimal_path(dag, dag_values, dag_grads, t_reach, x_reach, best_next_dag_id, tv=tv, reaching_eps=reaching_eps)

                # Combine solutions
                t_combined = np.concatenate([sol.t[:best_reach_index+1], next_sol.t])
                x_combined = np.concatenate([sol.y[:,:best_reach_index+1], next_sol.y], axis=1)
                sol = ode._ivp.ivp.OdeResult(t = t_combined, y = x_combined)
                dag_path = dag_path + next_dag_path
                switch_times = [t_reach] + next_switch_times

            # Else (eg. no more time, impossible reach), return the current RA solution

        return sol, dag_path, switch_times
        
    # Example start point in room3
    x_start = np.array([0.1, 0.1])
    t_start = -5.0
    sol, full_dag_path, switch_times = construct_optimal_path(dag_builder, dict_vars, value_tree_grads, t_start, x_start, dag_root, tv=False, reaching_eps=0.01)
    dag_ra_path = [i for i in full_dag_path if type(dag_builder.nodes[i]) in [DAGReachAvoid, DAGAvoid]]

    # Plot the optimal path on existing fig_rooms
    ax_rooms.plot(sol.y[0,:], sol.y[1,:], 'k-', linewidth=2, label='Optimal Path')
    ax_rooms.plot(x_start[0], x_start[1], 'o', markersize=6, label='', color='green')
    ax_rooms.plot(x_start[0], x_start[1], 'x', markersize=5, label='Start', color='black')
    dag_path_c = 1
    for switch_time in switch_times:
        switch_index = np.argmin(np.abs(sol.t - switch_time))
        tab_color = plt.get_cmap('tab10')(dag_ra_path[dag_path_c-1] % 10)
        ax_rooms.plot(sol.y[0,switch_index], sol.y[1,switch_index], 'o', markersize=6, label='switch: %d'%(dag_ra_path[dag_path_c-1]), color=tab_color, markeredgecolor='black', markeredgewidth=1)
        ax_rooms.text(sol.y[0,switch_index] + 0.05, sol.y[1,switch_index] + 0.02, "{:d}→{:d}".format(dag_ra_path[dag_path_c-1], dag_ra_path[dag_path_c-1]), color='black', fontsize=8)
        dag_path_c += 1

    ax_rooms.legend()
    fig_rooms.savefig("rooms_with_path.pdf", bbox_inches="tight")
    plt.close(fig_rooms)

if __name__ == "__main__":
    # with ipdb.launch_ipdb_on_exception():
    #     main()
    main()
