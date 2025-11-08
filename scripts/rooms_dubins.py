import os
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
from valtr.solver_utils import solve_dag_values
from valtr.control import construct_optimal_path

BASE_OUT_DIR = "results/rooms_dubins" # name for script
DIR_TAG = "onekey_test" # name for specific run
LOAD = False # whether to load existing results
    
class Car(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self,
                 max_acceleration=4.,
                 max_curvature=4.,
                 max_position_disturbance=0.,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_acceleration, -max_curvature]),
                                        jnp.array([max_acceleration, max_curvature]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_position_disturbance)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, v, q = state
        return jnp.array([v * jnp.cos(q), v * jnp.sin(q), 0., 0.])

    def control_jacobian(self, state, time):
        v = state[2]
        return jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., v],])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.],])

def main():

    os.makedirs('results', exist_ok=True)
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_OUT_DIR, DIR_TAG), exist_ok=True)
    os.chdir(os.path.join(BASE_OUT_DIR, DIR_TAG))

    # -------------------------------------------------------------------------------------------
    # Initialize dynamics, environment and task

    ## Define the grid
    lbs = np.array([-1.0, 0.0, -0.5, -np.pi])
    ubs = np.array([2.0, 1.0, 2.0, np.pi])
    grid_pad = np.array([0.2, 0.2, 0., 0.])
    # grid_nx, grid_ny, grid_nv, grid_nq = 201, 81, 11, 11
    grid_nx, grid_ny, grid_nv, grid_nq = 51, 21, 31, 31
    # grid_nx, grid_ny, grid_nv, grid_nq = 25, 11, 31, 31 # quick test

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lbs - grid_pad, ubs + grid_pad), [grid_nx, grid_ny, grid_nv, grid_nq],
        periodic_dims=3
    )

    bb_pos = np.array(grid.states)
    bb_X = bb_pos[:, :, 0, 0, 0]
    bb_Y = bb_pos[:, :, 0, 0, 1]

    grid_dict={"lbs": lbs, "ubs": ubs, "grid_pad": grid_pad, 
               "grid_nx": grid_nx, "grid_ny": grid_ny,
               "grid_nv": grid_nv, "grid_nq": grid_nq,
        "grid": grid, "bb_X": bb_X, "bb_Y": bb_Y, "grid_slice": np.s_[..., int(grid_nv/2), int(grid_nq/2)]}

    ## Define the rooms environment
    rooms_bc_dict = make_rooms(grid_dict)
    fig_rooms, ax_rooms = plot_rooms(rooms_bc_dict, 
                                     xmin=lbs[0]-grid_pad[0], 
                                     xmax=ubs[0]+grid_pad[0], 
                                     ymin=lbs[1]-grid_pad[1], 
                                     ymax=ubs[1]+grid_pad[1])

    ## Define the system dynamics
    dyn = Car()
    tf = 5.0
    ntimes = 4
    times = np.linspace(0.0, tf, ntimes+1)
    gamma = 0.9999
    # gamma = 1 # no discount -> bad control; just to check best satisfiability

    ## Define the task specification in TL
    if 'onekey' in DIR_TAG:
        task_source = "(!door1 U key1) && G( !walls ) && G( in_grid )" # 'onekey'
    elif 'twokey' in DIR_TAG:
        task_source = "(!door1 U key1) && (!door2 U key2) && G( !walls )" # 'twokey'
    elif 'threekey' in DIR_TAG:
        task_source = "(!door1 U key1) && (!door2 U key2) && F key3 && G( !walls )" # 'threekey'
    else:
        task_source = "(!door1 U key1) && G( !walls ) && G( in_grid )" # 'onekey'

    # -------------------------------------------------------------------------------------------
    # Parse and lower the task specification to a value tree DAG.
    logger.info("Generating the value tree DAG from logic...")
    print(f"Input task logic: {task_source}")

    lexer = TLLexer()
    tokens = list(lexer.tokenize(task_source))
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
    # Iterate through the DAG to solve all values.

    #     Input variables into the expression.
    #     Positive sdf is truthy.

    dict_predicates = {
        "key1": rooms_bc_dict["key1"],
        "door1": rooms_bc_dict["door1"],
        "key2": rooms_bc_dict["key2"],
        "door2": rooms_bc_dict["door2"],
        "key3": rooms_bc_dict["key3"],
        "walls": rooms_bc_dict["walls"],
        "in_grid": rooms_bc_dict["in_grid"],
    }

    if not LOAD:
        logger.info("Solving the value tree ...")
        value_tree_solution = solve_dag_values(dag_builder, dict_predicates, grid_dict, dyn, times, gamma)
        np.savez_compressed("value_tree_solution.npz", **{str(k): v for k, v in value_tree_solution.items()})
    else:
        logger.info("Loading presolved value tree ...")
        value_tree_solution_loaded = np.load("value_tree_solution.npz") 
        value_tree_solution = {}
        for k in value_tree_solution_loaded:
            value_tree_solution[int(k)] = value_tree_solution_loaded[k]

    # Final result is at dag_root
    bb_sdf_result = value_tree_solution[dag_root]

    # Plot the final result.
    fig, ax = plt.subplots(layout="constrained")
    cmap = "RdBu"
    norm = CenteredNorm()
    im = ax.contourf(bb_X, bb_Y, bb_sdf_result[:, :, int(grid_dict["grid_nv"]/2), int(grid_dict["grid_nq"]/2)], # middle slice
                     levels=25, cmap=cmap, norm=norm)
    ln = ax.contour(bb_X, bb_Y, bb_sdf_result[:, :, int(grid_dict["grid_nv"]/2), int(grid_dict["grid_nq"]/2)], # middle slice
                    levels=0, colors="black", linewidths=2)
    cbar = fig.colorbar(im, ax=ax)
    cbar.add_lines(ln)

    fig.suptitle("{}".format(task_source))
    fig.savefig("sol.pdf", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------------------------
    # Solve optimal path from the solved values
    logger.info("Solving optimal path(s)...")

    # Make dict of value tree gradients, assuming time-invariant for now
    value_tree_grads = {}
    for key, val in value_tree_solution.items():
        value_tree_grads[key] = grid.grad_values(val)
        # value_tree_grads[key] = [grid.grad_values(val[i,...]) for i in range(len(times))]
        
    # Example start point in room3
    # x_start = np.array([0.1, 0.1, 0.5, np.pi/2])
    # x_start = np.array([0.1, 0.1, 0.5, np.pi/4])
    x_start = np.array([0.1, 0.6, 0.5, 0.])
    t_start = -10.0
    sol, full_dag_path, switch_times = construct_optimal_path(
        dag_builder, 
        value_tree_solution, 
        value_tree_grads, 
        t_start, 
        x_start, 
        dag_root, 
        grid, 
        dyn, 
        times=times, 
        tv=False, 
        reaching_eps=0., 
        integration_method='jax'
    )
    dag_ra_path = [i for i in full_dag_path if type(dag_builder.nodes[i]) in [DAGReachAvoid, DAGAvoid]]
    print("Optimal path constructed,")
    print("  Full DAG path nodes: ", full_dag_path)
    print("  RA/A DAG path nodes: ", dag_ra_path)
    print("  Switch times: [{}]".format(", ".join([f"{st:2.2f}" for st in switch_times])))

    # Plot the optimal path on existing fig_rooms
    ax_rooms.plot(sol.y[0,:], sol.y[1,:], 'k-', linewidth=2, label='Optimal Path')
    ax_rooms.plot(x_start[0], x_start[1], 'o', markersize=6, label='', color='white')
    ax_rooms.plot(x_start[0], x_start[1], 'x', markersize=5, label='Start', color='black')
    dag_path_c = 1
    for st in switch_times:
        switch_index = np.argmin(np.abs(sol.t - st))
        tab_color = plt.get_cmap('tab10')(dag_path_c % 10)
        ax_rooms.plot(sol.y[0,switch_index], sol.y[1,switch_index], 'o', markersize=6, label='switch: %d'%(dag_ra_path[dag_path_c]), color=tab_color, markeredgecolor='black', markeredgewidth=1)
        ax_rooms.text(sol.y[0,switch_index] + 0.05, sol.y[1,switch_index] + 0.02, "{:d}→{:d}".format(dag_ra_path[dag_path_c-1], dag_ra_path[dag_path_c]), color='black', fontsize=8)
        dag_path_c += 1

    ax_rooms.legend()
    fig_rooms.savefig("rooms_with_path.pdf", bbox_inches="tight")
    plt.close(fig_rooms)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

# Utility functions

def sdf_aabb(pt: jnp.ndarray, center: jnp.ndarray, halfw_x: float, halfw_y: float):
    dx = jnp.abs(pt[0] - center[0]) - halfw_x
    dy = jnp.abs(pt[1] - center[1]) - halfw_y
    dx_clamped = jnp.maximum(dx, 0.0)
    dy_clamped = jnp.maximum(dy, 0.0)
    outside_dist = jnp.sqrt(dx_clamped**2 + dy_clamped**2)
    inside_dist = jnp.maximum(dx, dy)
    return outside_dist + inside_dist

def sdf_aabb_blcorner(pt: jnp.ndarray, bl_corner: jnp.ndarray, halfw_x: float, halfw_y: float):
    center = bl_corner + jnp.array([halfw_x, halfw_y])
    return sdf_aabb(pt, center, halfw_x, halfw_y)

def sdf_circle(pt: jnp.ndarray, center: jnp.ndarray, radius: float):
    return jnp.linalg.norm(pt - center) - radius

def make_rooms(grid_dict: hj.Grid) -> dict[str, np.ndarray]:

    lbs, ubs = grid_dict["lbs"], grid_dict["ubs"]
    # grid_pad = grid_dict["grid_pad"]
    grid_pad = [0,0,0,0] # no pad for sdf computation
    grid = grid_dict["grid"]

    # SDF for room 1: BL corner at (0, 0), width = height = 1
    def sdf_room1(pt):
        return sdf_aabb_blcorner(pt, jnp.array([0.0, 0.0]), 0.5, 0.5)

    # Room2 is to the right of room1. Also width = height = 1.
    def sdf_room2(pt):
        return sdf_aabb_blcorner(pt, jnp.array([1.0, 0.0]), 0.5, 0.5)

    # Room3 is to the left of room1. Also width = height = 1.
    def sdf_room3(pt):
        return sdf_aabb_blcorner(pt, jnp.array([-1.0, 0.0]), 0.5, 0.5)

    wall_thickness = 0.1

    # door1 is between room1 and room2. width=0.1, height of 1.0. Centered between the two rooms.
    def sdf_door12(pt):
        halfw_x = 0.05
        return sdf_aabb_blcorner(pt, jnp.array([0.95, -wall_thickness]), halfw_x, 0.5 + wall_thickness)

    # door2 is between room1 and room3. width=0.1, height of 1.0. Centered between the two rooms.
    def sdf_door13(pt):
        halfw_x = 0.05
        return sdf_aabb_blcorner(pt, jnp.array([0.0 - halfw_x, -wall_thickness]), halfw_x, 0.5 + wall_thickness)

    # wallB is below all three rooms. width=3.0, height=wall_thickness. Centered below the three rooms.
    def sdf_wallB(pt):
        return sdf_aabb_blcorner(pt, jnp.array([-1.0 - wall_thickness, -wall_thickness]), 1.5 + wall_thickness, wall_thickness / 2)

    # sdf_wallT is above all three rooms. width=3.0, height=wall_thickness. Centered above the three rooms.
    def sdf_wallT(pt):
        return sdf_aabb_blcorner(pt, jnp.array([-1.0 - wall_thickness, 1.0]), 1.5 + wall_thickness, wall_thickness / 2)

    # sdf_wallL is to the left of all three rooms. width=wall_thickness, height=1.0. Centered to the left of the three rooms.
    def sdf_wallL(pt):
        return sdf_aabb_blcorner(pt, jnp.array([-1.0 - wall_thickness, -wall_thickness]), wall_thickness / 2, 0.5 + wall_thickness)

    def sdf_wallR(pt):
        return sdf_aabb_blcorner(pt, jnp.array([2.0, -wall_thickness]), wall_thickness / 2, 0.5 + wall_thickness)

    def sdf_walls(pt):
        return jnp.min(jnp.array([
            sdf_wallB(pt),
            sdf_wallT(pt),
            sdf_wallL(pt),
            sdf_wallR(pt),
        ]))
    
    # sdf for grid limits for all dims
    def sdf_grid_limits(pt):
        sdf_xmin = pt[0] - (lbs[0] + grid_pad[0])
        sdf_xmax = (ubs[0] - grid_pad[0]) - pt[0]
        sdf_ymin = pt[1] - (lbs[1] + grid_pad[1])
        sdf_ymax = (ubs[1] - grid_pad[1]) - pt[1]
        sdf_vmin = pt[2] - (lbs[2] + grid_pad[2])
        sdf_vmax = (ubs[2] - grid_pad[2]) - pt[2]
        sdf_qmin = pt[3] - (lbs[3] + grid_pad[3])
        sdf_qmax = (ubs[3] - grid_pad[3]) - pt[3]
        return jnp.min(jnp.array([
            sdf_xmin, sdf_xmax,
            sdf_ymin, sdf_ymax,
            sdf_vmin, sdf_vmax,
            sdf_qmin, sdf_qmax,
        ]))
        
    # Key1 is in room1, Key2 is in room 2, Key3 is in room3.
    def sdf_key1(pt):
        return sdf_circle(pt, jnp.array([0.2, 0.6]), 0.05)

    def sdf_key2(pt):
        return sdf_circle(pt, jnp.array([1.75, 0.2]), 0.05)

    def sdf_key3(pt):
        return sdf_circle(pt, jnp.array([-0.6, 0.7]), 0.05)

    def jit_rep_vmap(fn, rep: int, *args):
        return np.array(rep_vmap(fn, rep=rep)(*args))
    
    bb_pos = np.array(grid.states)[:,:,0,0,:2] # only position affects value
    bb_sdf_room1 = -jit_rep_vmap(sdf_room1, 2, bb_pos)
    bb_sdf_room2 = -jit_rep_vmap(sdf_room2, 2, bb_pos)
    bb_sdf_room3 = -jit_rep_vmap(sdf_room3, 2, bb_pos)

    bb_sdf_door12 = -jit_rep_vmap(sdf_door12, 2, bb_pos)
    bb_sdf_door13 = -jit_rep_vmap(sdf_door13, 2, bb_pos)
    bb_sdf_walls = -jit_rep_vmap(sdf_walls, 2, bb_pos)

    bb_sdf_key1 = -jit_rep_vmap(sdf_key1, 2, bb_pos)
    bb_sdf_key2 = -jit_rep_vmap(sdf_key2, 2, bb_pos)
    bb_sdf_key3 = -jit_rep_vmap(sdf_key3, 2, bb_pos)

    bb_sdf_grid_limits = jit_rep_vmap(sdf_grid_limits, 4, np.array(grid.states)) # all states affect grid limits

    rooms_bc_dict = {
        "room1": bb_sdf_room1,
        "room2": bb_sdf_room2,
        "room3": bb_sdf_room3,
        "door1": bb_sdf_door12,
        "door2": bb_sdf_door13,
        "walls": bb_sdf_walls,
        "key1": bb_sdf_key1,
        "key2": bb_sdf_key2,
        "key3": bb_sdf_key3,
        "in_grid": bb_sdf_grid_limits,
    }

    # extend to 4d grid
    for k, v in rooms_bc_dict.items():
        if k == "in_grid":
            continue
        rooms_bc_dict[k] = jnp.broadcast_to(v[:, :, None, None], grid.shape)

    return rooms_bc_dict

def plot_rooms(rooms_bc_dict: dict[str, np.ndarray], xmin: float = -1.2, xmax: float = 2.2, ymin: float = -0.3, ymax: float = 1.2):
    
    def shade_supzero(ax_: plt.Axes, bb_sdf, color, alpha: float = 0.5):
        # Mask negative values
        masked = np.ma.array(bb_sdf, mask=bb_sdf < 0)

        ax_.imshow(
            masked.T,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            cmap=ListedColormap([color]),
            alpha=alpha,
            interpolation="nearest",
        )

    logger.info("Plotting SDFs...")
    fig_rooms, ax_rooms = plt.subplots(layout="constrained")
    ax_rooms.set_aspect("equal")

    # Shade inside the rooms.
    shade_supzero(ax_rooms, rooms_bc_dict["room1"][:,:,0,0], "C1", alpha=0.5)
    shade_supzero(ax_rooms, rooms_bc_dict["room2"][:,:,0,0], "C2", alpha=0.5)
    shade_supzero(ax_rooms, rooms_bc_dict["room3"][:,:,0,0], "C4", alpha=0.5)

    shade_supzero(ax_rooms, rooms_bc_dict["door1"][:,:,0,0], "C5", alpha=0.8)
    shade_supzero(ax_rooms, rooms_bc_dict["door2"][:,:,0,0], "C5", alpha=0.8)
    shade_supzero(ax_rooms, rooms_bc_dict["walls"][:,:,0,0], "k", alpha=0.8)

    shade_supzero(ax_rooms, rooms_bc_dict["key1"][:,:,0,0], "C6", alpha=0.8)
    shade_supzero(ax_rooms, rooms_bc_dict["key2"][:,:,0,0], "C6", alpha=0.8)
    shade_supzero(ax_rooms, rooms_bc_dict["key3"][:,:,0,0], "C6", alpha=0.8)

    fig_path = "rooms_sdf.pdf"
    fig_rooms.savefig(fig_path, bbox_inches="tight")
    return fig_rooms, ax_rooms

if __name__ == "__main__":
    # with ipdb.launch_ipdb_on_exception():
    #     main()
    main()
