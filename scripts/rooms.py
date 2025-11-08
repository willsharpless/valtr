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
import copy 

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
from valtr.control import construct_optimal_path, plot_optimal_path

BASE_OUT_DIR = "results/rooms" # name for script
DIR_TAG = "threekey" # name for specific run
LOAD = False  # whether to solve/load value tree

class Point(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, u_bd=1.0, d_bd=0.0, N=1, control_mode="max", disturbance_mode="min"):
        self.N = N
        self.dim = 2 * N
        control_space = hj.sets.Ball(jnp.array([0, 0]), u_bd)
        disturbance_space = hj.sets.Ball(jnp.array([0, 0]), d_bd)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.zeros_like(state)

    def control_jacobian(self, state, time):
        return jnp.eye(self.dim)

    def disturbance_jacobian(self, state, time):
        return jnp.eye(self.dim)

def main():

    os.makedirs('results', exist_ok=True)
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_OUT_DIR, DIR_TAG), exist_ok=True)
    os.chdir(os.path.join(BASE_OUT_DIR, DIR_TAG))
    os.makedirs("node_values", exist_ok=True)

    # -------------------------------------------------------------------------------------------
    # Initialize dynamics, environment and task

    ## Define the grid
    lbs = np.array([-1.0, 0.0])
    ubs = np.array([2.0, 1.0])
    grid_pad = np.array([0.2, 0.2])
    grid_nx = 501
    grid_ny = 201

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lbs - grid_pad, ubs + grid_pad), [grid_nx, grid_ny]
    )

    bb_pos = np.array(grid.states)
    bb_X = bb_pos[:, :, 0]
    bb_Y = bb_pos[:, :, 1]

    grid_dict={"lbs": lbs, "ubs": ubs, "grid_pad": grid_pad, "grid_nx": grid_nx, "grid_ny": grid_ny,
        "grid": grid, "bb_X": bb_X, "bb_Y": bb_Y, "grid_slice": (slice(None), slice(None))}

    ## Define the rooms environment
    rooms_bc_dict = make_rooms(grid)
    fig_rooms, ax_rooms = plot_rooms(rooms_bc_dict, 
                                     xmin=bb_X.min(), 
                                     xmax=bb_X.max(), 
                                     ymin=bb_Y.min(), 
                                     ymax=bb_Y.max())
    
    ## Define the system dynamics
    dyn = Point()
    tf = 5.0
    ntimes = 5
    times = np.linspace(0.0, tf, ntimes)
    gamma = 0.99999
    # gamma = 1 # no discount -> bad control; just to check best satisfiability

    ## Define the task specification in TL
    if 'onekey' in DIR_TAG:
        task_source = "(!door1 U key1) && G( !walls )" # 'onekey'
    elif 'twokey' in DIR_TAG:
        task_source = "(!door1 U key1) && (!door2 U key2) && G( !walls )" # 'twokey'
    elif 'threekey' in DIR_TAG:
        task_source = "(!door1 U key1) && (!door2 U key2) && F key3 && G( !walls )" # 'threekey'
    else:
        task_source = "(!door1 U key1) && G( !walls )" # 'onekey'

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

    # Plot the final result on a copy of fig_rooms
    fig_sol = copy.deepcopy(fig_rooms)
    ax_sol = fig_sol.axes[0]
    im = ax_sol.contourf(bb_X, bb_Y, bb_sdf_result, levels=25, cmap="RdBu", norm=CenteredNorm(), alpha=0.8)
    ln = ax_sol.contour(bb_X, bb_Y, bb_sdf_result, levels=0, colors="black", linewidths=2, alpha=0.8)
    divider = make_axes_locatable(ax_sol)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig_sol.colorbar(im, cax=cax)
    cbar.add_lines(ln)
    ax_sol.set_title("Value for [{}]".format(task_source))
    fig_sol.savefig("sol.pdf", bbox_inches="tight")
    plt.close(fig_sol)

    # -------------------------------------------------------------------------------------------
    # Solve optimal path from the solved values
    logger.info("Solving optimal path(s)...")

    # Make dict of value tree gradients, assuming time-invariant for now
    value_tree_grads = {}
    for key, val in value_tree_solution.items():
        value_tree_grads[key] = grid.grad_values(val)
        # value_tree_grads[key] = [grid.grad_values(val[i,...]) for i in range(len(times))]
        
    # Example start point in room3
    x_start = np.array([0.1, 0.1])
    t_start = -5.0
    sol, full_dag_path, switch_times = construct_optimal_path(dag_builder, value_tree_solution, value_tree_grads, t_start, x_start, dag_root, grid, dyn, times=times, tv=False, reaching_eps=0.01)
    dag_ra_path = [i for i in full_dag_path if type(dag_builder.nodes[i]) in [DAGReachAvoid, DAGAvoid]]
    print("Optimal path constructed,")
    print("  Full DAG path nodes: ", full_dag_path)
    print("  RA/A DAG path nodes: ", dag_ra_path)
    print("  Switch times: [{}]".format(", ".join([f"{st:2.2f}" for st in switch_times])))

    fig_rooms_path = plot_optimal_path(sol, dag_ra_path, switch_times, fig_base=fig_rooms)
    plt.close(fig_rooms_path)
    plt.close(fig_rooms)

    # ----------------------------------------------------------------------------
    logger.info("Complete.")
    print(f"See ./{BASE_OUT_DIR}/{DIR_TAG}/ for results.")

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

def make_rooms(grid: hj.Grid) -> dict[str, np.ndarray]:

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

    # Key1 is in room1, Key2 is in room 2, Key3 is in room3.
    def sdf_key1(pt):
        return sdf_circle(pt, jnp.array([0.2, 0.6]), 0.05)

    def sdf_key2(pt):
        return sdf_circle(pt, jnp.array([1.75, 0.2]), 0.05)

    def sdf_key3(pt):
        return sdf_circle(pt, jnp.array([-0.6, 0.7]), 0.05)

    def jit_rep_vmap(fn, rep: int, *args):
        return np.array(rep_vmap(fn, rep=rep)(*args))
    
    bb_pos = np.array(grid.states)
    bb_sdf_room1 = -jit_rep_vmap(sdf_room1, 2, bb_pos)
    bb_sdf_room2 = -jit_rep_vmap(sdf_room2, 2, bb_pos)
    bb_sdf_room3 = -jit_rep_vmap(sdf_room3, 2, bb_pos)

    bb_sdf_door12 = -jit_rep_vmap(sdf_door12, 2, bb_pos)
    bb_sdf_door13 = -jit_rep_vmap(sdf_door13, 2, bb_pos)
    bb_sdf_walls = -jit_rep_vmap(sdf_walls, 2, bb_pos)

    bb_sdf_key1 = -jit_rep_vmap(sdf_key1, 2, bb_pos)
    bb_sdf_key2 = -jit_rep_vmap(sdf_key2, 2, bb_pos)
    bb_sdf_key3 = -jit_rep_vmap(sdf_key3, 2, bb_pos)

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
    }

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
    shade_supzero(ax_rooms, rooms_bc_dict["room1"], "C1", alpha=0.5)
    shade_supzero(ax_rooms, rooms_bc_dict["room2"], "C2", alpha=0.5)
    shade_supzero(ax_rooms, rooms_bc_dict["room3"], "C4", alpha=0.5)

    shade_supzero(ax_rooms, rooms_bc_dict["door1"], "C5", alpha=0.8)
    shade_supzero(ax_rooms, rooms_bc_dict["door2"], "C5", alpha=0.8)
    shade_supzero(ax_rooms, rooms_bc_dict["walls"], "k", alpha=0.8)

    shade_supzero(ax_rooms, rooms_bc_dict["key1"], "C6", alpha=0.8)
    shade_supzero(ax_rooms, rooms_bc_dict["key2"], "C6", alpha=0.8)
    shade_supzero(ax_rooms, rooms_bc_dict["key3"], "C6", alpha=0.8)

    fig_path = "rooms_sdf.pdf"
    fig_rooms.savefig(fig_path, bbox_inches="tight")
    return fig_rooms, ax_rooms

if __name__ == "__main__":
    # with ipdb.launch_ipdb_on_exception():
    #     main()
    main()
