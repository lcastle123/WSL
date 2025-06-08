from constants import *
from dynamics import DubinsSkater, SleighSkater
from visualizations import *
from simulate import *
from controllers import *

# Define obstacle motion function - MOVED TO THE TOP
def get_opponent_position(i, t):
    """
    Get the position of opponent i at time t.
    For BRT calculation (t from 0 to -T):
    - At t=0: opponents are at their final positions (after moving)
    - At t=-T: opponents are at their initial positions
    """
    # Get initial positions
    initial_x = OPPONENT_X[i]
    initial_y = OPPONENT_Y[i]
    
    # Center of rink in y-dimension
    center_y = RINK_WIDTH / 2
    
    # Total movement amounts
    x_move = 10.0  # Total x movement in meters
    y_move = 5.0   # Maximum y movement
    
    # For BRT, at t=0, opponents should be at their "final" position
    # As t goes to -T, they move back to their initial position
    
    # Calculate movement factor (0 at t=0, 1 at t=-T)
    move_factor = (-t / T)
    
    # X position: start 10m to the left (at t=0), move right as t→-T
    final_x = initial_x - x_move
    new_x = final_x + (x_move * move_factor)
    
    # Y position: start at center (at t=0), move outward as t→-T
    y_diff = center_y - initial_y  # Direction from initial to center
    final_y = initial_y + y_diff * (y_move / jnp.maximum(jnp.abs(y_diff), y_move))
    new_y = final_y - (y_diff * move_factor * (y_move / jnp.maximum(jnp.abs(y_diff), y_move)))
    
    return new_x, new_y

# Define time_dependent_avoid_set BEFORE using it
def time_dependent_avoid_set(t, x):
    """Time-dependent avoid set for moving opponents"""
    # Calculate obstacle positions at this time point
    obstacle_distances = []
    for i in range(NUM_OPS):
        ox, oy = get_opponent_position(i, t)
        # Create grid of distances
        distances = jnp.sqrt((grid.states[...,0] - ox)**2 + 
                            (grid.states[...,1] - oy)**2) - OPPONENT_R
        obstacle_distances.insert(0,distances)
    # Combine with rink boundary (unchanged)
    phi_avoid_t = jnp.min(jnp.stack(obstacle_distances), axis=0)
    phi_avoid_t = jnp.minimum(phi_avoid_t, rink_boundary)
    
    return jnp.maximum(x, -phi_avoid_t)

load_values = True
VIZ = False
# DYNAMICS = "SleighSkater"
DYNAMICS = "DubinsSkater"
#controller_type='Nominal'
#controller_type="SafeController"
controller_type="LeastRestrictive"

if DYNAMICS == "DubinsSkater":
    dynamics = DubinsSkater(control_mode="min", disturbance_mode="max")
    s0 = jnp.array([2, 1+RINK_WIDTH/2, jnp.pi, 0])  # Initial state (x, y, theta, v)
elif DYNAMICS == "SleighSkater": 
    dynamics = SleighSkater()
    s0 = jnp.array([0,  RINK_WIDTH - 5, 0, 0, 0, 0, 0, 0, 0])  # Initial state

grid = dynamics.grid

# Compute signed-distance fields for all obstacles efficiently
obstacle_distances = [
    jnp.sqrt((grid.states[...,0] - OPPONENT_X[i])**2 +
             (grid.states[...,1] - OPPONENT_Y[i])**2) - OPPONENT_R
    for i in range(NUM_OPS)
]

rink_boundary = jnp.minimum(grid.states[...,0], 
                            jnp.minimum(RINK_LENGTH - grid.states[...,0], 
                            jnp.minimum(grid.states[...,1], 
                            RINK_WIDTH - grid.states[...,1])))

# 2. signed-distance fields
phi_avoid = jnp.min(jnp.stack(obstacle_distances), axis=0)  # <0 inside obstacles
phi_avoid = jnp.minimum(phi_avoid, rink_boundary)  # <0 outside rink boundary
phi_target = jnp.sqrt((grid.states[...,0]-GOAL_X)**2 +
                      (grid.states[...,1]-GOAL_Y)**2) - GOAL_R  # <0 inside goal

# Initial avoid set (t=0)
obstacle_distances = []
for i in range(NUM_OPS):
    # Use initial positions at t=0
    ox, oy = get_opponent_position(i, 0)
    distances = jnp.sqrt((grid.states[...,0] - ox)**2 + 
                        (grid.states[...,1] - oy)**2) - OPPONENT_R
    obstacle_distances.append(distances)

# Initial avoid set
phi_avoid = jnp.min(jnp.stack(obstacle_distances), axis=0)
phi_avoid = jnp.minimum(phi_avoid, rink_boundary)

# Initial values for reach-avoid
initial_values = jnp.maximum(phi_target, -phi_avoid)


# Solver Settings
times = np.linspace(0, -T, VALS_NT, endpoint=True)
# With this time-dependent version:
solver_settings = hj.SolverSettings.with_accuracy(
    "high", 
    hamiltonian_postprocessor=lambda x: jnp.minimum(x, 0),
    value_postprocessor=time_dependent_avoid_set  # Use the time-dependent version
)

if load_values:
    # Load precomputed values if available
    values = np.load('outputs/values_moving.npy')  # Use a different filename
else:
    values = hj.solve(solver_settings, dynamics, grid, times, initial_values)
    np.save('outputs/values_moving.npy', values)  # Save with a different name

if VIZ:
    # To create a time series of snapshots
    for i in range(0, len(times), 5):
        visualize_xy_slice(DYNAMICS, values, grid, times, time_idx=i, 
                         theta=0, v=5, omega=0,
                         get_opponent_position_fn=get_opponent_position,
                         save_path=f'brt_t{i}.png')
    theta_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[2]) - 0))
    v_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[3]) - 10))
    omega_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[4]) - 0))   

    # Create a grid of visualizations across different velocities
    for v_val in [0, 2, 4]:
        visualize_xy_slice(DYNAMICS, values, grid, times, time_idx=-1,
                           theta=0, v=v_val, omega=0,
                           save_path=f'brt_v{v_val}.png')

    # Create visualization of how obstacles move at different time steps
    for i in range(0, len(times), 10):
        # Create a special visualization showing obstacle positions at this time
        visualize_xy_slice_with_moving_obstacles(DYNAMICS, values, grid, times, 
                                               time_idx=i, theta=0, v=5,
                                               save_path=f'brt_moving_t{i}.png',
                                               get_opponent_position_fn=get_opponent_position)
    save_values_gif(values[:,:,:,11,-1], grid, times, 
                  get_opponent_position_fn=get_opponent_position,
                  save_path='values.gif') 

ctlr = Controller(values, dynamics, grid, controller_type=controller_type, solver_settings=solver_settings, times=times)

dyn = lambda s, u: dynamics.f_cl(s, u)

# When calling simulate, pass the get_opponent_position function
#s_history, u_history, t_history = simulate(s0, ctlr, T, f=dyn, get_opponent_position_fn=get_opponent_position)

#visualize_xy_slice(DYNAMICS, values, grid, times, time_idx=-1, 
#                  theta=0, v=10, omega=0, 
#                  show_plot=True, s_history=s_history, save_path='trajectory.png')

#visualize_state_history(s_history, t_history, save_path='state_history.png')
#visualize_control_history(u_history, t_history[:-1], save_path='control_history.png')

# After simulation completes and you have s_history, u_history, t_history
# Create animation with value function background
# animate_trajectory_with_value_function(
#     s_history=s_history,
#     t_history=t_history,
#     values=values,
#     grid=grid,
#     times=times,
#     dyn_type=DYNAMICS,
#     theta=0,
#     v=5,
#     omega=0,
#     show_value_function=True,
#     title=f"Hockey Player Trajectory with {DYNAMICS} Dynamics",
#     get_opponent_position_fn=get_opponent_position,
#     save_path="trajectory_animation.gif",
#     fps=5
# )

# Run 100 simulations to test the controller
num_simulations = 1
num_successful_simulations = 0
ttg_running_sum = 0.0  # Time to goal running sum
for i in range(num_simulations):
    # Set the initial state for each simulation with some randomness
    s0 = jnp.array([
        np.random.uniform(0, RINK_LENGTH/5),
        np.random.uniform(0, RINK_WIDTH),
        np.random.uniform(-jnp.pi, jnp.pi),
        np.random.uniform(0, 10)  # Random initial velocity
    ])
    s_history, u_history, t_history, time_to_goal = simulate(s0, ctlr, T, f=dyn, get_opponent_position_fn=get_opponent_position)
    # Keep track of the number of successful simulations
    if check_goal(s_history[-1]):
        num_successful_simulations += 1
        ttg_running_sum += time_to_goal
        print(f"Simulation {i+1} reached the goal in {time_to_goal:.2f} seconds.")
        print(f"Average time to goal so far: {ttg_running_sum / num_successful_simulations:.2f} seconds.")
        print(f"Success rate: {num_successful_simulations / (i + 1) * 100:.2f}%")
        if i % 10 == 0:
            animate_trajectory_with_value_function(
            s_history=s_history,
            t_history=t_history,
            values=values,
            grid=grid,
            times=times,
            dyn_type=DYNAMICS,
            theta=0,
            v=5,
            omega=0,
            show_value_function=True,
            title=f"Hockey Player Trajectory with {DYNAMICS} Dynamics - Simulation {i+1}",
            get_opponent_position_fn=get_opponent_position,
            save_path=f"trajectory_animation_simulation_nom{i+1}.mp4",
            fps=10
            )
    else:
        animate_trajectory_with_value_function(
        s_history=s_history,
        t_history=t_history,
        values=values,
        grid=grid,
        times=times,
        dyn_type=DYNAMICS,
        theta=0,
        v=5,
        omega=0,
        show_value_function=True,
        title=f"Hockey Player Trajectory with {DYNAMICS} Dynamics - Simulation {i+1}",
        get_opponent_position_fn=get_opponent_position,
        save_path=f"trajectory_animation_simulation_nom{i+1}.mp4",
        fps=10
        )
        print(f"Simulation {i+1} did not reach the goal.")


# Print the number of successful simulations
print(f"Number of successful simulations: {num_successful_simulations}/{num_simulations}")
