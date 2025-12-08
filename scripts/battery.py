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
import datetime

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

MODEL = "POINT" # "POINT" or "DUBINS"
BASE_OUT_DIR = "results/battery_{}".format(MODEL) # name for script
DIR_TAG = "r1_recharge_flow_v1_smooth" # name for specific run
LOAD = False # whether to load existing results

## Define the task specification in TL
# later, we map logic -> target/obstacle (doors, keys, walls, grid limits)
if 'r1' in DIR_TAG:
    TASK_SOURCE = "F tg && G in_grid"
elif 'r2' in DIR_TAG:
    TASK_SOURCE = "F (tg && F tg) && G in_grid"
elif 'r3' in DIR_TAG:
    TASK_SOURCE = "F (tg && F (tg && F tg)) && G in_grid"
else:
    TASK_SOURCE = "G F tg"

## Define environment dynamics
class PointBattery(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, 
                 u_bd=5., 
                 d_bd=0., 
                 charge_loss_rate=0.,
                 recharge_rate=0.,
                 port_center=jnp.array([0., 0.]),
                 port_radius=0.2,
                 flow_x=0.,
                 flow_y=0.,
                 control_mode="max", 
                 disturbance_mode="min"):
        # self.N = N
        # self.dim = 2 * N
        self.charge_loss_rate = charge_loss_rate
        self.recharge_rate = recharge_rate
        self.port_center = port_center
        self.port_radius = port_radius
        self.flow_x = flow_x
        self.flow_y = flow_y
        control_space = hj.sets.Ball(jnp.array([0, 0]), u_bd)
        disturbance_space = hj.sets.Ball(jnp.array([0, 0]), d_bd)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    # def open_loop_dynamics(self, state, time):
    #     _, _, charge = state
    #     return jnp.array([self.flow_x, 
    #                       self.flow_y, 
    #                       -self.charge_loss_rate * (charge > 0.) + \
    #                         self.recharge_rate * (jnp.linalg.norm(state[:2] - self.port_center) < self.port_radius) * (charge < 1.)
    #                       ])
    
    def open_loop_dynamics(self, state, time): # smooth approx
        _, _, charge = state
        return jnp.array([self.flow_x, 
                          self.flow_y, 
                          -self.charge_loss_rate * (1/(1 + jnp.exp(5 - 50 * charge))) + \
                            self.recharge_rate * jnp.exp((jnp.log(0.1/2)/(self.port_radius**2)) * jnp.linalg.norm(state[:2] - self.port_center) ** 2) * (1/(1 + jnp.exp(5 + 50 * (charge - 1.))))
                          ])

    def control_jacobian(self, state, time):
        _, _, charge = state
        return jnp.array([[jnp.abs(charge), 0.], [0., jnp.abs(charge)], [0., 0.]])

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((3, 2))
    
class DubinsBattery(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self,
                 velocity=1.,
                 max_curvature=8.,
                 max_position_disturbance=0.,
                 charge_loss_rate=0.1,
                 recharge_rate=0.1,
                 port_center = jnp.array([0., 0.]),
                 port_radius=0.2,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.velocity = velocity
        self.charge_loss_rate = charge_loss_rate
        self.recharge_rate = recharge_rate
        self.port_center = port_center
        self.port_radius = port_radius
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_curvature]),
                                        jnp.array([ max_curvature]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_position_disturbance)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        _, _, theta, charge = state
        return jnp.array([self.velocity * jnp.cos(theta), 
                          self.velocity * jnp.sin(theta), 
                          0., 
                          -self.charge_loss_rate * jnp.sign(jnp.max(charge, 0.)) + \
                            self.recharge_rate * (jnp.linalg.norm(state[:2] - self.port_center) < self.port_radius)
                          ])

    def control_jacobian(self, state, time):
        _, _, _, charge = state
        return jnp.array([[0.], [0.], [jnp.abs(charge) * 1.], [0.]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])

def main():

    os.makedirs('results', exist_ok=True)
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_OUT_DIR, DIR_TAG), exist_ok=True)
    os.chdir(os.path.join(BASE_OUT_DIR, DIR_TAG))
    os.makedirs("node_values", exist_ok=True)

    # -------------------------------------------------------------------------------------------
    # Initialize dynamics, environment and task

    X_LEFT = -1.0
    X_RIGHT = 1.0
    Y_BOTTOM = 0.0
    Y_TOP = 2.0
    TARGET_CENTER = jnp.array([-0.5, 1.5])
    TARGET_RADIUS = 0.2
    PORT_CENTER = jnp.array([0.5, 0.5])
    PORT_RADIUS = 0.2

    U_MAX = 1.5
    FLOW_X = 0.0
    FLOW_Y = 0.0 if "flow" not in DIR_TAG else 0.5 
    # ^must > (BOX HEIGHT)/TIME to capture inf-time worst-case
    CHARGE_LOSS_RATE = 0.35
    # ^want >= 1/TIME to observe dead battery trajectories
    RECHARGE_RATE = 2. if "recharge" in DIR_TAG else 0.
    # ^determines time between goal recurrence

    ## Define the grid
    if MODEL == "POINT":
        dyn = PointBattery(u_bd=U_MAX, charge_loss_rate=CHARGE_LOSS_RATE, recharge_rate=RECHARGE_RATE,
                           port_center=PORT_CENTER, port_radius=PORT_RADIUS,
                           flow_x=FLOW_X, flow_y=FLOW_Y)
        TF = 1.0

        lbs = np.array([X_LEFT, Y_BOTTOM, -0.0])
        ubs = np.array([X_RIGHT, Y_TOP, 1.0])
        grid_pad = np.array([0.1, 0.1, 0.05])
        grid_lens = [grid_nx, grid_ny, grid_nc] = 101, 101, 51
        # grid_nx, grid_ny, grid_nq = 201, 81, 61
        # grid_nx, grid_ny, grid_nq = 101, 41, 31 
        
    elif MODEL == "DUBINS":
        dyn = DubinsBattery(u_bd=U_MAX, charge_loss_rate=CHARGE_LOSS_RATE, recharge_rate=RECHARGE_RATE,
                           port_center=PORT_CENTER, port_radius=PORT_RADIUS,
                           flow_x=FLOW_X, flow_y=FLOW_Y)
        TF = 3.0

        lbs = np.array([X_LEFT, Y_BOTTOM, -np.pi, 0.0])
        ubs = np.array([X_RIGHT, Y_TOP, np.pi, 1.0])
        grid_pad = np.array([0.1, 0.1, 0., 0.0])
        grid_lens = [grid_nx, grid_ny, grid_nq, grid_nc] = 401, 161, 121, 51
        # grid_nx, grid_ny, grid_nq, grid_nc = 201, 81, 61, 26
        # grid_nx, grid_ny, grid_nq, grid_nc = 101, 41, 31, 16
        # grid_nx, grid_ny, grid_nq, grid_nc = 51, 21, 15, 8
        
    else:
        raise ValueError("Unknown MODEL {}".format(MODEL))

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(lbs - grid_pad, ubs + grid_pad), grid_lens,
        periodic_dims=2 if MODEL == "DUBINS" else None
    )

    charge_slice_ix = -5
    pos = np.array(grid.states)
    grid_X = pos[:, :, charge_slice_ix, 0] if MODEL == "POINT" else pos[:, :, int(grid_nq/2), charge_slice_ix, 0]
    grid_Y = pos[:, :, charge_slice_ix, 1] if MODEL == "POINT" else pos[:, :, int(grid_nq/2), charge_slice_ix, 1]

    grid_dict={
        "lbs": lbs, 
        "ubs": ubs, 
        "grid_pad": grid_pad, 
        "grid_nx": grid_nx, 
        "grid_ny": grid_ny, 
        "grid_nq": grid_nq if MODEL == "DUBINS" else 1, 
        "grid_nc": grid_nc,
        "grid": grid, 
        "grid_X": grid_X, 
        "grid_Y": grid_Y, 
        "grid_slice": np.s_[..., charge_slice_ix] if MODEL == "POINT" \
                    else np.s_[..., int(grid_nq/2), charge_slice_ix],
        "target_center": TARGET_CENTER,           
        "target_radius": TARGET_RADIUS,
        "port_center": PORT_CENTER,
        "port_radius": PORT_RADIUS,
        "flow_x": FLOW_X,
        "flow_y": FLOW_Y,
        "charge_loss_rate": CHARGE_LOSS_RATE,
        "recharge_rate": RECHARGE_RATE,
    }

    ## Define the rooms environment
    rooms_bc_dict = make_rooms(grid_dict)
    fig_rooms, ax_rooms = plot_rooms(rooms_bc_dict, grid_dict)

    ## Define the system dynamics
    ntimes = 4
    times = np.linspace(0.0, TF, ntimes+1)
    GAMMA = 0.99999
    # gamma = 1 # no discount -> bad control (just for satisfiability)

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
        "tg": rooms_bc_dict["target"],
        "port": rooms_bc_dict["port"],
        "in_grid": rooms_bc_dict["in_grid"],
    }

    if not LOAD:
        logger.info("Solving the value tree ...")
        value_tree_solution = solve_dag_values(value_tree_dag, dict_predicates, grid_dict, dyn, times, GAMMA)
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
                         value_tree_solution[dag_root][grid_dict["grid_slice"]],
                        levels=25, cmap="RdBu", norm=CenteredNorm(), alpha=0.8)
    ln = ax_sol.contour(grid_X, grid_Y, 
                        value_tree_solution[dag_root][grid_dict["grid_slice"]],
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
    x_start = np.array([0., 0.1, 1.])
    # x_start = np.array([0.1, 0.6, 0.])
    t_start = -TF
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

    fig_rooms_path = plot_optimal_path(sol, dag_ra_path, switch_times, fig_base=fig_rooms,
                                       color_path=True, color_path_type='state', color_path_state_ix=-1,
                                       color_path_lb=0, color_path_ub=1, color_path_state_label='Battery',
                                       )
    plt.close(fig_rooms_path)
    plt.close(fig_rooms)

    # -------------------------------------------------------------------------------------------
    # Batch solve for gif
    print("Solving batch paths for gif...")

    # grid of points
    X_start = np.array(np.meshgrid(
        np.linspace(-0.9, 0.9, 5),
        np.linspace(0.1, 1.9, 5),
    )).reshape(2, -1).T  # shape (N, 2)
    
    # concatenate with orientations
    if MODEL == "DUBINS":
        X_start = np.concatenate([
            X_start,
            # all facing towards key1
            np.arctan2(TARGET_CENTER[1] - X_start[:,1], TARGET_CENTER[0] - X_start[:,0])[:, None]
        ], axis=1)  # shape (N, 3)

    # concatenate with battery charge
    X_start = np.concatenate([
        X_start,
        1. * jnp.ones((X_start.shape[0], 1))
    ], axis=1)  # shape (N, 3)
    
    # t_start = -2.0 if MODEL == "POINT" else -3.0
    t_start = -TF
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
            skip_frames=1,  # Show more frames for detailed view
            grid_dict=grid_dict,
        )

    # ----------------------------------------------------------------------------        
    logger.info("Complete.")
    write_config(TF,
            GAMMA,
            U_MAX,
            FLOW_X,
            FLOW_Y,
            CHARGE_LOSS_RATE,
            RECHARGE_RATE,
            X_LEFT,
            X_RIGHT,
            Y_BOTTOM,
            Y_TOP,
            TARGET_CENTER,
            TARGET_RADIUS,
            PORT_CENTER,
            PORT_RADIUS)
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
    grid_pad = [0,0,0,0] # no pad for sdf computation
    grid = grid_dict["grid"]
    
    # sdf for grid limits for all dims (other than battery charge)
    def sdf_grid_limits(pt):
        # sdf_xmin = pt[0] - (lbs[0] + grid_pad[0])
        # sdf_xmax = (ubs[0] - grid_pad[0]) - pt[0]
        # sdf_ymin = pt[1] - (lbs[1] + grid_pad[1])
        # sdf_ymax = (ubs[1] - grid_pad[1]) - pt[1]
        sdf_xmin = pt[0] - lbs[0]
        sdf_xmax = ubs[0] - pt[0]
        sdf_ymin = pt[1] - lbs[1]
        sdf_ymax = ubs[1] - pt[1]
        if MODEL == "POINT":
            return jnp.min(jnp.array([
                sdf_xmin, sdf_xmax,
                sdf_ymin, sdf_ymax,
            ]))
        elif MODEL == "DUBINS":
            sdf_qmin = pt[2] - (lbs[2] + grid_pad[2])
            sdf_qmax = (ubs[2] - grid_pad[2]) - pt[2]
            return jnp.min(jnp.array([
                sdf_xmin, sdf_xmax,
                sdf_ymin, sdf_ymax,
                sdf_qmin, sdf_qmax,
            ]))
        
    def sdf_target(pt):
        return sdf_circle(pt, grid_dict["target_center"], grid_dict["target_radius"])

    def sdf_port(pt):
        return sdf_circle(pt, grid_dict["port_center"], grid_dict["port_radius"])
    # just for plotting, not used in value computation

    def jit_rep_vmap(fn, rep: int, *args):
        return np.array(rep_vmap(fn, rep=rep)(*args))

    pos = np.array(grid.states)[:,:,0,:2] if MODEL == "POINT" else np.array(grid.states)[:,:,0,0,:2]
    # only position affects value

    bb_sdf_target = -jit_rep_vmap(sdf_target, 2, pos)
    bb_sdf_port = -jit_rep_vmap(sdf_port, 2, pos)

    if MODEL == "POINT":
        bb_sdf_grid_limits = jit_rep_vmap(sdf_grid_limits, 3, np.array(grid.states)) # all states affect grid limits
    elif MODEL == "DUBINS":
        bb_sdf_grid_limits = jit_rep_vmap(sdf_grid_limits, 4, np.array(grid.states)) # all states affect grid limits
    else:
        raise ValueError("Unknown MODEL {}".format(MODEL))
    
    rooms_bc_dict = {
        "target": bb_sdf_target,
        "port": bb_sdf_port,
        "in_grid": bb_sdf_grid_limits,
    }

    # extend to 3d/4d grid
    for k, v in rooms_bc_dict.items():
        if k == "in_grid":
            continue
        if MODEL == "POINT":
            rooms_bc_dict[k] = jnp.broadcast_to(v[:, :, None], grid.shape)
        elif MODEL == "DUBINS":
            rooms_bc_dict[k] = jnp.broadcast_to(v[:, :, None, None], grid.shape)
        else:
            raise ValueError("Unknown MODEL {}".format(MODEL))

    return rooms_bc_dict

def plot_batch_trajectories_gif(
    batch_results: dict,
    base_figure: plt.Figure,
    filename: str = "batch_trajectories.gif",
    fps: int = 60,
    skip_frames: int = 1,
    max_frames: int = 300,
    trail_length: int = 30,
    grid_dict: dict = None,
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
    
    batch_size, total_steps, state_dim = states.shape
    
    # Define colors for different DAG nodes
    # dag_color_map = {
    #     38: '#ff7f0e',  # Orange
    #     19: '#2ca02c',  # Green
    #     10:  '#9467bd',  # Purple
    #     7:  '#34495e',  # Dark Gray
    #     -1: '#95a5a6'   # Light Gray (terminal)
    # }
    
    # def get_dag_color(dag_id):
    #     return dag_color_map.get(dag_id, '#bdc3c7')  # Default light gray
    color_map = plt.get_cmap('RdYlGn')
    norm = plt.Normalize(vmin=0, vmax=1)
    
    # Initialize scatter plots for trajectories
    scatters = []
    trails = []
    
    marker = (3, 0, 0) if MODEL == "DUBINS" else 'o'
    for i in range(batch_size):
        # Current position scatter (larger, more visible)
        scatter = ax.scatter([], [], s=35, alpha=0.9, zorder=10, edgecolors='black', linewidth=1, marker=marker)
        scatters.append(scatter)
        
        # Trail line
        trail, = ax.plot([], [], alpha=0.6, linewidth=1, zorder=5)
        trails.append(trail)

    # Text elements for main plot
    time_text = ax.text(0.85, 0.9, "", transform=ax.transAxes, zorder=10,
                            verticalalignment='top', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", alpha=0.8, facecolor="white"))

    # add colorbar for battery charge
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Battery', fontsize=8)
    # add tick for each agent
    states_unique = np.unique(states[:,0,-1])
    cbar.set_ticks(states_unique)
    cbar.set_ticklabels([''] * len(states_unique))
    cbar.ax.tick_params(
        direction='out',
        length=10,
        width=2,
        colors='black',
        labelsize=8
    )

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
            # color = get_dag_color(current_dag_id)
            color = color_map(norm(np.clip(states[i, step, -1], 0.0, 1.0)))  # battery charge

            # Update current position (if in bounds)
            if (x > grid_dict["lbs"][0] and x < grid_dict["ubs"][0] and y > grid_dict["lbs"][1] and y < grid_dict["ubs"][1]):
                scatters[i].remove()
                marker = (3, 0, np.degrees(theta)-90) if MODEL == "DUBINS" else 'o'
                scatters[i] = ax.scatter([x], [y], s=35, alpha=0.9, zorder=10, edgecolors='black', linewidth=1, 
                                        marker=marker, color=color)
            else:
                scatters[i].set_color("black")
                scatters[i].set_alpha(0.)
                trails[i].set_alpha(0.)

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
        
        # Update ticks
        states_unique = np.unique(states[:,step,-1])
        cbar.set_ticks(states_unique)
        cbar.set_ticklabels([''] * len(states_unique))
        
        # Update text displays
        current_time = times[0, step] if not np.isnan(times[0, step]) else 0.0
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

def plot_rooms(rooms_bc_dict: dict[str, np.ndarray], grid_dict: dict = None) -> tuple[plt.Figure, plt.Axes]:

    x_min = grid_dict["lbs"][0] - grid_dict["grid_pad"][0]
    x_max = grid_dict["ubs"][0] + grid_dict["grid_pad"][0]
    y_min = grid_dict["lbs"][1] - grid_dict["grid_pad"][1]
    y_max = grid_dict["ubs"][1] + grid_dict["grid_pad"][1]
    
    def shade_supzero(ax_: plt.Axes, bb_sdf, color, alpha: float = 0.5, label: str = ""):
        # Mask negative values
        masked = np.ma.array(bb_sdf, mask=bb_sdf < 0)

        ax_.imshow(
            masked.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
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
    shade_supzero(ax_rooms, rooms_bc_dict["target"][grid_dict["grid_slice"]], "C1", alpha=0.9, label="Target")
    shade_supzero(ax_rooms, rooms_bc_dict["port"][grid_dict["grid_slice"]], "C8", alpha=0.9, label="Port")

    # Plot contour for recharge function
    recharge_function = lambda state: grid_dict["recharge_rate"] * jnp.exp((jnp.log(0.1/2)/(grid_dict["port_radius"]**2)) * jnp.linalg.norm(state - grid_dict["port_center"]) ** 2)
    X, Y = grid_dict["grid_X"], grid_dict["grid_Y"]
    # recharge_value = recharge_function(jnp.stack([X, Y], axis=-1)) # this yields zero-dim array (wrong, should be two-dim)
    # recharge_value = np.array(recharge_function(jnp.stack([X, Y], axis=-1))) # still wrong (0D array)
    recharge_value = np.array(jnp.vectorize(recharge_function, signature='(n)->()')(jnp.stack([X, Y], axis=-1)))

    # make custom colormap with alpha channel
    from matplotlib.colors import LinearSegmentedColormap
    colors = ListedColormap(["C0"])(np.linspace(0, 1, 256))
    colors[:, -1] = np.linspace(0, 1., 256)  # vary alpha
    custom_alpha_cmap = LinearSegmentedColormap.from_list("custom_alpha", colors)
    ax_rooms.contourf(X, Y, recharge_value, levels=50, cmap=custom_alpha_cmap)

    # shade_supzero(ax_rooms, rooms_bc_dict["room1"][:,:,0], "C1", alpha=0.3, label="Room 1")
    # shade_supzero(ax_rooms, rooms_bc_dict["room2"][:,:,0], "C2", alpha=0.3, label="Room 2")
    # shade_supzero(ax_rooms, rooms_bc_dict["room3"][:,:,0], "C4", alpha=0.3, label="Room 3")

    # shade_supzero(ax_rooms, rooms_bc_dict["key1"][:,:,0], "C1", alpha=0.9, label="Key 1")
    # shade_supzero(ax_rooms, rooms_bc_dict["key2"][:,:,0], "C2", alpha=0.9, label="Key 2")
    # shade_supzero(ax_rooms, rooms_bc_dict["key3"][:,:,0], "C4", alpha=0.9, label="Key 3")

    # shade_supzero(ax_rooms, rooms_bc_dict["door1"][:,:,0], "C5", alpha=0.9, label="Door 1")
    # shade_supzero(ax_rooms, rooms_bc_dict["door2"][:,:,0], "C6", alpha=0.9, label="Door 2")
    shade_supzero(ax_rooms, -rooms_bc_dict["in_grid"][grid_dict["grid_slice"]], "C7", alpha=0.9, label="Bounds")

    # add quiver corresponding to flow field
    if grid_dict["flow_x"] != 0.0 or grid_dict["flow_y"] != 0.0:
        sparsity = 32
        buffer = 15
        color="C0"
        X = grid_dict["grid_X"][buffer:-buffer:sparsity, buffer:-buffer:sparsity]
        Y = grid_dict["grid_Y"][buffer:-buffer:sparsity, buffer:-buffer:sparsity]
        U = np.zeros_like(X) + grid_dict["flow_x"]
        V = np.zeros_like(Y) + grid_dict["flow_y"]
        ax_rooms.quiver(X, Y, U, V, color=color, alpha=0.2, scale=5, zorder=3, label="Flow")
        X = grid_dict["grid_X"][buffer+sparsity//2:-buffer:sparsity, buffer+sparsity//2:-buffer:sparsity]
        Y = grid_dict["grid_Y"][buffer+sparsity//2:-buffer:sparsity, buffer+sparsity//2:-buffer:sparsity]
        U = np.zeros_like(X) + grid_dict["flow_x"]
        V = np.zeros_like(Y) + grid_dict["flow_y"]
        ax_rooms.quiver(X, Y, U, V, color=color, alpha=0.2, scale=5, zorder=3)

    ax_rooms.legend(frameon=True, facecolor="white", framealpha=0.8, ncol=2, loc="lower left")

    from matplotlib import font_manager

    fig_path = "rooms_sdf.pdf"
    ax_rooms.set_title("    BATTERY-{} ".format(MODEL), loc="left", fontsize=10, fontweight="bold")
    ax_rooms.text(
        0.35, 1.025, TASK_SOURCE,
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

def write_config(TF,
                 GAMMA,
                 U_MAX,
                 FLOW_X,
                 FLOW_Y,
                 CHARGE_LOSS_RATE,
                 RECHARGE_RATE,
                 X_LEFT,
                 X_RIGHT,
                 Y_BOTTOM,
                 Y_TOP,
                 TARGET_CENTER,
                 TARGET_RADIUS,
                 PORT_CENTER,
                 PORT_RADIUS,
                 config_filename: str = "config.txt"):
        
    # write config of all constants
    with open(config_filename, "w") as f:
        f.write(f"DATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"BASE_OUT_DIR: {BASE_OUT_DIR}\n")
        f.write(f"DIR_TAG: {DIR_TAG}\n")
        f.write(f"MODEL: {MODEL}\n")
        f.write(f"TASK_SOURCE: {TASK_SOURCE}\n")
        f.write(f"TF: {TF}\n")
        f.write(f"GAMMA: {GAMMA}\n")
        f.write(f"U_MAX: {U_MAX}\n")
        f.write(f"FLOW_X: {FLOW_X}\n")
        f.write(f"FLOW_Y: {FLOW_Y}\n")
        f.write(f"CHARGE_LOSS_RATE: {CHARGE_LOSS_RATE}\n")
        f.write(f"RECHARGE_RATE: {RECHARGE_RATE}\n")
        f.write(f"X_LEFT: {X_LEFT}\n")
        f.write(f"X_RIGHT: {X_RIGHT}\n")
        f.write(f"Y_BOTTOM: {Y_BOTTOM}\n")
        f.write(f"Y_TOP: {Y_TOP}\n")
        f.write(f"TARGET_CENTER: {TARGET_CENTER}\n")
        f.write(f"TARGET_RADIUS: {TARGET_RADIUS}\n")
        f.write(f"PORT_CENTER: {PORT_CENTER}\n")
        f.write(f"PORT_RADIUS: {PORT_RADIUS}\n")

if __name__ == "__main__":
    # with ipdb.launch_ipdb_on_exception():
    #     main()
    main()
