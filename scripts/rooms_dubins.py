import os
import hj_reachability as hj
import hj_reachability.dynamics as dynamics
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import CenteredNorm, ListedColormap
import numpy as np
from loguru import logger
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
from valtr.control import construct_optimal_path, plot_optimal_path, \
    construct_optimal_path_batch, construct_optimal_path_batch_fast, construct_optimal_path_batch_auto
import time

plt.style.use("seaborn-v0_8-darkgrid")

BASE_OUT_DIR = "results/rooms_dubins" # name for script
DIR_TAG = "threekey" # name for specific run, 'threekey_dense_g99999_t3'
LOAD = True # whether to load existing results

## Define the task specification in TL
# later, we map logic -> target/obstacle (doors, keys, walls, grid limits)
if 'threekey' in DIR_TAG:
    TASK_SOURCE = "(!d1 U k1) && (!d2 U k2) && F k3 && G( !w ) && G( g )" # 'threekey'
elif 'twokey' in DIR_TAG:
    TASK_SOURCE = "(!d1 U k1) && (!d2 U k2) && G( !w ) && G( g )" # 'twokey'
else:
    TASK_SOURCE = "(!d1 U k1) && G( !w ) && G( g )" # 'onekey'

## Define environment dynamics
class Dubins(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self,
                 velocity=1.,
                 max_curvature=8.,
                 max_position_disturbance=0.,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.velocity = velocity
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_curvature]),
                                        jnp.array([ max_curvature]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_position_disturbance)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, theta = state
        return jnp.array([self.velocity * jnp.cos(theta), self.velocity * jnp.sin(theta), 0.])

    def control_jacobian(self, state, time):
        return jnp.array([[0.], [0.], [1.],])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[1., 0.], [0., 1.], [0., 0.],])

def main():

    os.makedirs('results', exist_ok=True)
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_OUT_DIR, DIR_TAG), exist_ok=True)
    os.chdir(os.path.join(BASE_OUT_DIR, DIR_TAG))
    os.makedirs("node_values", exist_ok=True)

    # -------------------------------------------------------------------------------------------
    # Initialize dynamics, environment and task

    ## Define the grid
    lbs = np.array([-1.0, 0.0, -np.pi])
    ubs = np.array([2.0, 1.0, np.pi])
    grid_pad = np.array([0.2, 0.2, 0.])
    grid_nx, grid_ny, grid_nq = 401, 161, 121
    # grid_nx, grid_ny, grid_nq = 201, 81, 61
    # grid_nx, grid_ny, grid_nq = 101, 41, 31

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lbs - grid_pad, ubs + grid_pad), [grid_nx, grid_ny, grid_nq],
        periodic_dims=2
    )

    pos = np.array(grid.states)
    grid_X = pos[:, :, 0, 0]
    grid_Y = pos[:, :, 0, 1]

    grid_dict={"lbs": lbs, "ubs": ubs, "grid_pad": grid_pad, 
               "grid_nx": grid_nx, "grid_ny": grid_ny, "grid_nq": grid_nq,
                "grid": grid, "grid_X": grid_X, "grid_Y": grid_Y, 
                "grid_slice": np.s_[..., int(grid_nq/2)]}

    ## Define the rooms environment
    rooms_bc_dict = make_rooms(grid_dict)
    fig_rooms, ax_rooms = plot_rooms(rooms_bc_dict, 
                                     xmin=lbs[0]-grid_pad[0], 
                                     xmax=ubs[0]+grid_pad[0], 
                                     ymin=lbs[1]-grid_pad[1], 
                                     ymax=ubs[1]+grid_pad[1])

    ## Define the system dynamics
    dyn = Dubins()
    tf = 3.0
    ntimes = 4
    times = np.linspace(0.0, tf, ntimes+1)
    gamma = 0.99999
    # gamma = 1 # no discount -> bad control; just to check best satisfiability

    # -------------------------------------------------------------------------------------------
    # Parse and lower the task specification to a value tree DAG.
    logger.info("Generating the value tree DAG from logic...")
    print(f"Input task logic: {TASK_SOURCE}")

    lexer = TLLexer()
    tokens = list(lexer.tokenize(TASK_SOURCE))
    ast = TLParser(tokens).parse()

    # AST -> IR
    ir = IRBuilder()
    lowerer = Lowerer(builder=ir)
    ir_root_id = lowerer.lower(ast)

    passes = [PassFinallyToUntil, PassCombineGloballySegments]
    for p_cls in passes:
        p = p_cls(ir)
        ir_root_id, ir = p.run(ir_root_id)

    # IR -> DAG
    value_tree_dag, dag_root = lower_ir_to_dag(ir, ir_root_id)

    # Perform constant folding.
    passes = [PassFoldConstBool]
    for p_cls in passes:
        p = p_cls(value_tree_dag)
        dag_root, value_tree_dag, changed = p.run(dag_root)

    # Visualize the DAG.
    dot_dag = visualize_dag(value_tree_dag, dag_root, filename="value_tree_dag", view=True)

    # -------------------------------------------------------------------------------------------
    # Iterate through the DAG to solve all values.

    #     Input variables into the expression.
    #     Positive sdf is truthy.

    dict_predicates = {
        "k1": rooms_bc_dict["key1"],
        "d1": rooms_bc_dict["door1"],
        "k2": rooms_bc_dict["key2"],
        "d2": rooms_bc_dict["door2"],
        "k3": rooms_bc_dict["key3"],
        "w": rooms_bc_dict["walls"],
        "g": rooms_bc_dict["in_grid"],
    }

    if not LOAD:
        logger.info("Solving the value tree ...")
        value_tree_solution = solve_dag_values(value_tree_dag, dict_predicates, grid_dict, dyn, times, gamma)
        np.savez_compressed("value_tree_solution.npz", **{str(k): v for k, v in value_tree_solution.items()})
    else:
        logger.info("Loading presolved value tree ...")
        value_tree_solution_loaded = np.load("value_tree_solution.npz") 
        value_tree_solution = {}
        for k in value_tree_solution_loaded:
            value_tree_solution[int(k)] = value_tree_solution_loaded[k]

    # Plot the final result
    fig_sol = copy.deepcopy(fig_rooms)
    ax_sol = fig_sol.axes[0]
    im = ax_sol.contourf(grid_X, grid_Y, 
                         value_tree_solution[dag_root][:, :, int(grid_dict["grid_nq"]/2)], # middle slice
                        levels=25, cmap="RdBu", norm=CenteredNorm(), alpha=0.8)
    ln = ax_sol.contour(grid_X, grid_Y, 
                        value_tree_solution[dag_root][:, :, int(grid_dict["grid_nq"]/2)], # middle slice
                        levels=0, colors="black", linewidths=2, alpha=0.8)
    divider = make_axes_locatable(ax_sol)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig_sol.colorbar(im, cax=cax)
    cbar.add_lines(ln)
    ax_sol.set_title("Value")
    ax_sol.get_legend().remove()
    fig_sol.savefig("solution_values.pdf", bbox_inches="tight")
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
    x_start = np.array([0.1, 0.1, 0.])
    # x_start = np.array([0.1, 0.6, 0.])
    t_start = -10.0
    sol, full_dag_path, switch_times = construct_optimal_path(
        value_tree_dag, 
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
    dag_ra_path = [i for i in full_dag_path if type(value_tree_dag.nodes[i]) in [DAGReachAvoid, DAGAvoid]]
    print("Optimal path constructed,")
    print("  Full DAG path nodes: ", full_dag_path)
    print("  RA/A DAG path nodes: ", dag_ra_path)
    print("  Switch times: [{}]".format(", ".join([f"{st:2.2f}" for st in switch_times])))

    fig_rooms_path = plot_optimal_path(sol, dag_ra_path, switch_times, fig_base=fig_rooms)
    plt.close(fig_rooms_path)
    plt.close(fig_rooms)

    # -------------------------------------------------------------------------------------------
    # Batch solve for gif
    print("Solving batch paths for gif...")

    # grid of points in room 1
    X_start = np.array(np.meshgrid(
        np.linspace(0.1, 0.9, 5),
        np.linspace(0.05, 0.9, 5),
    )).reshape(2, -1).T  # shape (N, 2)
    t_start = -10.0
    # concatenate with orientations
    X_start = np.concatenate([
        X_start,
        # np.zeros((X_start.shape[0], 1))  # all facing right
        # all facing towards key1
        np.arctan2(0.6 - X_start[:,1], 0.2 - X_start[:,0])[:, None]
    ], axis=1)  # shape (N, 3)
    
    results_batch = construct_optimal_path_batch_auto(
        value_tree_dag, 
        value_tree_solution, 
        value_tree_grads, 
        t_start, 
        X_start, 
        dag_root, 
        grid,
        dyn, 
        times=times, 
        tv=False, 
        reaching_eps=0.01, 
        integration_method='jax',
        use_fast_version=False  # Use the regular batch version
    )

    if 'trajectory_history' in results_batch:
        plot_batch_trajectories_gif(
            results_batch,
            fig_rooms,
            filename=f"batch_trajectories.gif",
            fps=60,
            skip_frames=1  # Show more frames for detailed view
        )

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

def make_rooms(grid_dict: hj.Grid) -> dict[str, np.ndarray]:

    lbs, ubs = grid_dict["lbs"], grid_dict["ubs"]
    # grid_pad = grid_dict["grid_pad"]
    grid_pad = [0,0,0] # no pad for sdf computation
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
        sdf_qmin = pt[2] - (lbs[2] + grid_pad[2])
        sdf_qmax = (ubs[2] - grid_pad[2]) - pt[2]
        return jnp.min(jnp.array([
            sdf_xmin, sdf_xmax,
            sdf_ymin, sdf_ymax,
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
    
    pos = np.array(grid.states)[:,:,0,:2] # only position affects value
    bb_sdf_room1 = -jit_rep_vmap(sdf_room1, 2, pos)
    bb_sdf_room2 = -jit_rep_vmap(sdf_room2, 2, pos)
    bb_sdf_room3 = -jit_rep_vmap(sdf_room3, 2, pos)

    bb_sdf_door12 = -jit_rep_vmap(sdf_door12, 2, pos)
    bb_sdf_door13 = -jit_rep_vmap(sdf_door13, 2, pos)
    bb_sdf_walls = -jit_rep_vmap(sdf_walls, 2, pos)

    bb_sdf_key1 = -jit_rep_vmap(sdf_key1, 2, pos)
    bb_sdf_key2 = -jit_rep_vmap(sdf_key2, 2, pos)
    bb_sdf_key3 = -jit_rep_vmap(sdf_key3, 2, pos)

    bb_sdf_grid_limits = jit_rep_vmap(sdf_grid_limits, 3, np.array(grid.states)) # all states affect grid limits

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

    # extend to 3d grid
    for k, v in rooms_bc_dict.items():
        if k == "in_grid":
            continue
        rooms_bc_dict[k] = jnp.broadcast_to(v[:, :, None], grid.shape)

    return rooms_bc_dict

def plot_batch_trajectories_gif(
    batch_results: dict,
    base_figure: plt.Figure,
    filename: str = "batch_trajectories.gif",
    fps: int = 60,
    skip_frames: int = 1,
    max_frames: int = 300,
    trail_length: int = 30,
):
    """
    Create an animated GIF showing the evolution of batch trajectories through the rooms environment.
    
    Args:
        batch_results: Results from construct_optimal_path_batch containing trajectory_history
        base_figure: Mpl figure to use as background
        filename: Output filename for the GIF
        fps: Frames per second for the animation
        skip_frames: Skip every N frames to reduce file size
        xmin, xmax, ymin, ymax: Plot bounds
    """
    print(f"Creating trajectory evolution GIF: {filename}")
    
    # Extract trajectory data
    fig = copy.deepcopy(base_figure)
    ax = fig.axes[0]
    ax.set_title("Value - Optimal Flow", fontsize=12)
    ax.get_legend().remove()
    trajectory_history = batch_results['trajectory_history']
    states = trajectory_history['states']  # Shape: (batch_size, time_steps, state_dim)
    times = trajectory_history['times']    # Shape: (batch_size, time_steps)
    dag_ids = trajectory_history['dag_ids'] # Shape: (batch_size, time_steps)

    # trajectory with longest time duration
    traj_durations = np.nanmax(times, axis=1) - np.nanmin(times, axis=1)
    traj_longest_ix = np.argmax(traj_durations)
    print(f"  Longest trajectory index: {traj_longest_ix}, duration: {traj_durations[traj_longest_ix]:.2f}s")
    
    batch_size, total_steps, state_dim = states.shape
    
    # Define colors for different DAG nodes
    dag_color_map = {
        38: '#ff7f0e',  # Orange
        19: '#2ca02c',  # Green
        10:  '#9467bd',  # Purple
        7:  '#34495e',  # Dark Gray
        -1: '#95a5a6'   # Light Gray (terminal)
    }
    
    def get_dag_color(dag_id):
        return dag_color_map.get(dag_id, '#bdc3c7')  # Default light gray
    
    # Initialize scatter plots for trajectories
    scatters = []
    trails = []
    
    for i in range(batch_size):
        # Current position scatter (larger, more visible)
        scatter = ax.scatter([], [], s=35, alpha=0.9, zorder=10, edgecolors='black', linewidth=1, marker=(3, 0, 0))
        scatters.append(scatter)
        
        # Trail line
        trail, = ax.plot([], [], alpha=0.6, linewidth=1, zorder=5)
        trails.append(trail)

    # Text elements for main plot
    time_text = ax.text(0.85, 0.9, "", transform=ax.transAxes, zorder=10,
                            verticalalignment='top', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", alpha=0.8, facecolor="white"))

    def animate(frame):
        """Animation function called for each frame"""
        step = frame * skip_frames
        if step >= total_steps:
            step = total_steps - 1
            
        # Count trajectories in each DAG state
        dag_counts = {}
        
        # Update each trajectory
        for i in range(batch_size):
            # Get current state and DAG ID
            x, y = states[i, step, 0], states[i, step, 1]
            theta = states[i, step, 2]
            current_dag_id = dag_ids[i, step]
            current_time = times[i, step]
                
            dag_counts[current_dag_id] = dag_counts.get(current_dag_id, 0) + 1
            
            # Get color for current DAG node
            color = get_dag_color(current_dag_id)
            
            # Update current position
            scatters[i].remove()
            scatters[i] = ax.scatter([x], [y], s=35, alpha=0.9, zorder=10, edgecolors='black', linewidth=1, 
                                     marker=(3, 0, np.degrees(theta)-90), color=color)

            # Update trail (last N points)
            trail_start = max(0, step - trail_length)
            trail_x = states[i, trail_start:step+1, 0]
            trail_y = states[i, trail_start:step+1, 1]
            
            # Remove NaN values from trail
            valid_trail = ~(np.isnan(trail_x) | np.isnan(trail_y))
            if np.any(valid_trail):
                trail_x = trail_x[valid_trail]
                trail_y = trail_y[valid_trail]
                trails[i].set_data(trail_x, trail_y)
                # trails[i].set_color(color)
                trails[i].set_color('k')
            else:
                trails[i].set_data([], [])
        
        # Update text displays
        current_time = times[traj_longest_ix, step] if not np.isnan(times[traj_longest_ix, step]) else 0.0
        time_text.set_text(f"Time: {current_time:.2f}")

        # remove axes
        ax.tick_params(
            bottom=False,  # hides tick marks
            left=False,
            labelbottom=False,  # hides tick labels
            labelleft=False
        )
        ax.grid(True)

        return scatters + trails + [time_text]

    # Create animation
    if total_steps // skip_frames > max_frames:
        skip_frames = (total_steps + max_frames - 1) // max_frames
        total_frames = total_steps // skip_frames
    else:
        total_frames = total_steps // skip_frames
    print(f"Creating {total_frames} frames from {total_steps} simulation steps...")
    
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=1000//fps, 
        blit=False, repeat=True
    )
    
    # Save as GIF
    anim.save(filename, writer='pillow', fps=fps, dpi=300)
    print(f"GIF saved: {filename}")
    
    # Print file size
    file_size = os.path.getsize(filename) / 1024 / 1024  # MB
    print(f"   File size: {file_size:.1f} MB")
    
    plt.close(fig)
    return anim

def plot_rooms(rooms_bc_dict: dict[str, np.ndarray], xmin: float = -1.2, xmax: float = 2.2, ymin: float = -0.3, ymax: float = 1.2):
    
    def shade_supzero(ax_: plt.Axes, bb_sdf, color, alpha: float = 0.5, label: str = ""):
        # Mask negative values
        masked = np.ma.array(bb_sdf, mask=bb_sdf < 0)

        ax_.imshow(
            masked.T,
            origin="lower",
            extent=[xmin, xmax, ymin, ymax],
            cmap=ListedColormap([color]),
            alpha=alpha,
            interpolation="nearest",
            zorder=4,
        )
        # add square marker legend entry if label:
        if label:
            ax_.scatter([], [], marker="s", color=color, alpha=alpha, label=label)

    logger.info("Plotting SDFs...")
    fig_rooms, ax_rooms = plt.subplots(layout="constrained")
    ax_rooms.set_aspect("equal")

    # Shade inside the rooms.
    # shade_supzero(ax_rooms, rooms_bc_dict["room1"][:,:,0], "C1", alpha=0.3, label="Room 1")
    # shade_supzero(ax_rooms, rooms_bc_dict["room2"][:,:,0], "C2", alpha=0.3, label="Room 2")
    # shade_supzero(ax_rooms, rooms_bc_dict["room3"][:,:,0], "C4", alpha=0.3, label="Room 3")

    shade_supzero(ax_rooms, rooms_bc_dict["key1"][:,:,0], "C1", alpha=0.9, label="Key 1")
    shade_supzero(ax_rooms, rooms_bc_dict["key2"][:,:,0], "C2", alpha=0.9, label="Key 2")
    shade_supzero(ax_rooms, rooms_bc_dict["key3"][:,:,0], "C4", alpha=0.9, label="Key 3")

    shade_supzero(ax_rooms, rooms_bc_dict["door1"][:,:,0], "C5", alpha=0.9, label="Door 1")
    shade_supzero(ax_rooms, rooms_bc_dict["door2"][:,:,0], "C6", alpha=0.9, label="Door 2")
    shade_supzero(ax_rooms, rooms_bc_dict["walls"][:,:,0], "k", alpha=0.9, label="Walls")

    ax_rooms.legend(frameon=True, facecolor="white", framealpha=0.8, ncol=2, loc="lower left")

    from matplotlib import font_manager

    fig_path = "rooms_sdf.pdf"
    ax_rooms.set_title("      ROOMS-DUBINS ", loc="left", fontsize=10, fontweight="bold")
    ax_rooms.text(
        0.275, 1.05, TASK_SOURCE,
        fontfamily="monospace",
        transform=ax_rooms.transAxes,
        ha="left", va="center",
        fontsize=9,
    )

    fig_rooms.savefig(fig_path, bbox_inches="tight")
    
    # clear figure descriptors for reuse
    for artist in ax_rooms.get_children():
        if hasattr(artist, 'get_offsets') and len(artist.get_offsets()) == 0:
            artist.remove()
    
    ax_rooms.title.set_text("")
    ax_rooms.set_title("")

    for txt in ax_rooms.texts[:]:  # Use slice copy to avoid modification during iteration
        txt.remove()

    return fig_rooms, ax_rooms

if __name__ == "__main__":
    # with ipdb.launch_ipdb_on_exception():
    #     main()
    main()
