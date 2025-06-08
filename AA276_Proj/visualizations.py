import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on your system
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import os

from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from constants import GOAL_X, GOAL_Y, GOAL_R, RINK_LENGTH, RINK_WIDTH, OPPONENT_X, OPPONENT_Y, OPPONENT_R, NUM_OPS, T

# Get the absolute path to the project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a cache for opponent trajectories to ensure consistent visualization
OPPONENT_TRAJECTORY_CACHE = {}

def get_opponent_trajectory(get_opponent_position_fn, time_points):
    """
    Pre-computes opponent trajectories and caches them for consistent visualization
    
    Args:
        get_opponent_position_fn: Function that returns (x,y) given opponent index and time
        time_points: Array of time points to evaluate
        
    Returns:
        Dictionary mapping opponent index to arrays of [x,y] positions
    """
    global OPPONENT_TRAJECTORY_CACHE
    
    # Create a unique key based on the time points
    cache_key = f"{id(get_opponent_position_fn)}_{hash(tuple(time_points))}"
    
    # Return cached result if available
    if cache_key in OPPONENT_TRAJECTORY_CACHE:
        return OPPONENT_TRAJECTORY_CACHE[cache_key]
    # Make an array of opponent y positions as per the constant
    op_yis = np.array([OPPONENT_Y[j] for j in range(NUM_OPS)])
    # Compute trajectories for all opponents
    trajectories = {}
    for i in range(NUM_OPS):
        positions = []
        for t in time_points:
            x, y = get_opponent_position_fn(i, t)
            x = x + 10
            y = y + 5 * jnp.sign(op_yis[i] - RINK_WIDTH / 2)
            positions.append([x, y])
        trajectories[i] = np.array(positions)
    
    # Cache the result
    OPPONENT_TRAJECTORY_CACHE[cache_key] = trajectories
    return trajectories

def save_values_gif(values, grid, times, save_path=None):
    """
    args:
        values: ndarray with shape [
                len(times),
                len(grid.coordinate_vectors[0]),
                len(grid.coordinate_vectors[1])
            ]
        grid: hj.Grid
        times: ndarray with shape [len(times)]
        save_path: Optional path to save the gif (default: outputs/values.gif)
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "values.gif")
    else:
        # If relative path is given, make it absolute
        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)
            
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vbar = 3
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'$V(x, {times[0]:3.2f})$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    value_plot = ax.pcolormesh(
        grid.coordinate_vectors[0], grid.coordinate_vectors[1],
        np.clip(values[0].T, -vbar, vbar),   
        cmap='RdBu_r'
)
    plt.colorbar(value_plot, ax=ax)
    global value_contour
    value_contour = ax.contour(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[0].T,
        levels=0,
        colors='k'
    )

    def update(i):
        ax.set_title(f'$V(x, {times[i]:3.2f})$')
        value_plot.set_array(values[i].T)
        global value_contour
        value_contour.remove()
        value_contour = ax.contour(
            grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            values[i].T,
            levels=0,
            colors='k'
        )
        return ax
    anim = FuncAnimation(
        fig=fig,
        func=update,
        frames=np.arange(len(times)),
        interval=int(1000*(-times[1]))
    )
    with tqdm(total=len(times)) as anim_pbar:
        anim.save(filename=save_path, writer='pillow', progress_callback=lambda i, n: anim_pbar.update(1))
    print(f'SAVED GIF TO: {save_path}')
    plt.close()

def visualize_xy_slice(dyn_type, values, grid, times, time_idx, theta=0, v=None, omega=None,
                     title=None, save_path=None, show_plot=False, s_history=None, 
                     get_opponent_position_fn=None):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if dyn_type == "DubinsSkater":
        # Get the correct dimensions for Dubins model
        print(f"Value function shape: {values.shape}")
        
        # For 4D DubinsSkater: find theta_idx and v_idx
        theta_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[2]) - theta))
        
        # Use v=0 if not specified
        if v is None:
            v = 0
        v_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[3]) - v))
        
        # Extract a properly shaped 2D slice for visualization
        value_slice = values[time_idx, :, :, theta_idx, v_idx]
        
        # Get coordinate vectors for x and y dimensions
        x_coords = grid.coordinate_vectors[0]
        y_coords = grid.coordinate_vectors[1]
        
        # Set plot title
        if title is None:
            theta_val = grid.coordinate_vectors[2][theta_idx]
            v_val = grid.coordinate_vectors[3][v_idx]
            title = f'Value Function at t={times[time_idx]:.2f}, θ={theta_val:.2f}, v={v_val:.2f}'
    
    elif dyn_type == "SleighSkater":
        # For SleighSkater: state is [p1, p2, theta, x, y, a, b, v_a, v_b]
        # The x-y coordinates are at dimensions 3 and 4
        theta_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[2]) - theta))
        a_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[5]) - 0))  # Default to a=0
        b_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[6]) - 0))  # Default to b=0
        v_a_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[7]) - v))  # Use v for v_a 
        v_b_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[8]) - omega))  # Use omega for v_b
        
        # Extract a 2D slice for the xy plane (indices 3 and 4)
        # Use multi-dimensional indexing for the 9D grid
        value_slice = values[time_idx, 2, 2, theta_idx, :, :, a_idx, b_idx, v_a_idx, v_b_idx]
        
        # Get coordinate vectors for x and y dimensions
        x_coords = grid.coordinate_vectors[3]  # x is at index 3
        y_coords = grid.coordinate_vectors[4]  # y is at index 4
        
        # Set plot title
        if title is None:
            title = f'Value Function at t={times[time_idx]:.2f}, SleighSkater Model'
    
    else:
        raise ValueError(f"Unknown dynamics type: {dyn_type}")
    
    ax.set_title(title)
    
    # Create the color mesh
    vbar = 3  # Value range for color scale
    mesh = ax.pcolormesh(
        x_coords, y_coords, value_slice.T,
        cmap='RdBu_r', vmin=-vbar, vmax=+vbar
    )
    fig.colorbar(mesh, ax=ax)
    
    # Add contour for the zero level set
    contour = ax.contour(
        x_coords, y_coords, value_slice.T,
        levels=0, colors='k', linewidths=2
    )
    
    # Add labels and legend
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')

    # Set equal axis
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, RINK_LENGTH)
    ax.set_ylim(0, RINK_WIDTH)
    
    # Add hockey rink elements
    for i in range(NUM_OPS):
        # Get time-dependent opponent position if function provided
        if get_opponent_position_fn:
            ox, oy = get_opponent_position_fn(i, times[time_idx])
        else:
            # Default to static position
            ox, oy = OPPONENT_X[i], OPPONENT_Y[i]
        
        opponent = plt.Circle((ox, oy), OPPONENT_R, 
                            color='red', alpha=0.5, 
                            label='Opponent' if i == 0 else None)
        ax.add_patch(opponent)
    
    goal = plt.Circle((GOAL_X, GOAL_Y), GOAL_R, color='green', alpha=0.5, label='Goal Area')
    ax.add_patch(goal)
    
    rink = plt.Rectangle((0, 0), RINK_LENGTH, RINK_WIDTH, fill=False, edgecolor='blue', linestyle='--')
    ax.add_patch(rink)

    # Plot trajectory if provided
    if s_history is not None:
        # Check if s_history is a single state (1D) or multiple states (2D)
        if len(np.shape(s_history)) == 1:
            # Single state - just plot a point
            plt.plot(s_history[0], s_history[1], 'o', markersize=5, 
                    label='Skater Position', color='orange')
        else:
            # Multiple states - plot the trajectory
            plt.plot(s_history[:, 0], s_history[:, 1], 'o-', markersize=2, 
                    label='Skater Trajectory', color='orange')

    ax.legend(loc='best')    
    
    # Save if requested
    if save_path is not None:
        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax

def visualize_xy_slice_with_moving_obstacles(dyn_type, values, grid, times, time_idx, 
                                          theta=0, v=0, get_opponent_position_fn=None,
                                          save_path=None, show_plot=False):
    # Similar to visualize_xy_slice but with time-dependent opponent positions
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if dyn_type == "DubinsSkater":
        # Get the correct dimensions for Dubins model
        print(f"Value function shape: {values.shape}")
        
        # For 4D DubinsSkater: find theta_idx and v_idx
        theta_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[2]) - theta))
        
        # Use v=0 if not specified
        if v is None:
            v = 0
        v_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[3]) - v))
        
        # Extract a properly shaped 2D slice for visualization
        value_slice = values[time_idx, :, :, theta_idx, v_idx]
        
        # Get coordinate vectors for x and y dimensions
        x_coords = grid.coordinate_vectors[0]
        y_coords = grid.coordinate_vectors[1]
        
        # Set plot title
        theta_val = grid.coordinate_vectors[2][theta_idx]
        v_val = grid.coordinate_vectors[3][v_idx]
        title = f'Value Function at t={times[time_idx]:.2f}, θ={theta_val:.2f}, v={v_val:.2f}'
    
    elif dyn_type == "SleighSkater":
        # For SleighSkater: state is [p1, p2, theta, x, y, a, b, v_a, v_b]
        # The x-y coordinates are at dimensions 3 and 4
        theta_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[2]) - theta))
        a_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[5]) - 0))  # Default to a=0
        b_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[6]) - 0))  # Default to b=0
        v_a_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[7]) - v))  # Use v for v_a 
        v_b_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[8]) - v))  # Use omega for v_b
        
        # Extract a 2D slice for the xy plane (indices 3 and 4)
        # Use multi-dimensional indexing for the 9D grid
        value_slice = values[time_idx, 2, 2, theta_idx, :, :, a_idx, b_idx, v_a_idx, v_b_idx]
        
        # Get coordinate vectors for x and y dimensions
        x_coords = grid.coordinate_vectors[3]  # x is at index 3
        y_coords = grid.coordinate_vectors[4]  # y is at index 4
        
        # Set plot title
        title = f'Value Function at t={times[time_idx]:.2f}, SleighSkater Model'
    
    else:
        raise ValueError(f"Unknown dynamics type: {dyn_type}")
    
    ax.set_title(title)
    
    # Create the color mesh
    vbar = 3  # Value range for color scale
    mesh = ax.pcolormesh(
        x_coords, y_coords, value_slice.T,
        cmap='RdBu_r', vmin=-vbar, vmax=+vbar
    )
    fig.colorbar(mesh, ax=ax)
    
    # Add contour for the zero level set
    contour = ax.contour(
        x_coords, y_coords, value_slice.T,
        levels=0, colors='k', linewidths=2
    )
    
    # Add labels and legend
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')

    # Set equal axis
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, RINK_LENGTH)
    ax.set_ylim(0, RINK_WIDTH)
    
    # Add hockey rink elements
    rink_center_y = RINK_WIDTH / 2
    for i in range(NUM_OPS):
        # For static visualization, show opponents moved part-way to center
        y_diff = rink_center_y - OPPONENT_Y[i]
        new_y = OPPONENT_Y[i] + y_diff * 0.5  # Show moved 50% of the way
        
        opponent = plt.Circle((OPPONENT_X[i], new_y), OPPONENT_R, 
                            color='red', alpha=0.5)
        ax.add_patch(opponent)
    
    goal = plt.Circle((GOAL_X, GOAL_Y), GOAL_R, color='green', alpha=0.5, label='Goal Area')
    ax.add_patch(goal)
    
    rink = plt.Rectangle((0, 0), RINK_LENGTH, RINK_WIDTH, fill=False, edgecolor='blue', linestyle='--')
    ax.add_patch(rink)

    # Plot obstacles at their position for this time
    t = times[time_idx]
    for i in range(NUM_OPS):
        ox, oy = get_opponent_position_fn(i, t) if get_opponent_position_fn else (OPPONENT_X[i], OPPONENT_Y[i])
        
        opponent = plt.Circle((ox, oy), OPPONENT_R, color='red', alpha=0.5,
                             label='Opponent' if i == 0 else "")
        ax.add_patch(opponent)
    
    # Save if requested
    if save_path is not None:
        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax

def visualize_state_history(s_history, time, save_path=None):
    # Visualize the state history of each state using a for loop
    num_states = s_history.shape[1]
    fig, axs = plt.subplots(num_states, 1, figsize=(10, 2 * num_states))
    for i in range(num_states):
        axs[i].plot(time, s_history[:, i], label=f'State {i+1}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel(f'State {i+1}')
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.show()
    return fig, axs

def visualize_control_history(u_history, time, save_path=None):
    # Visualize the control history of each control input using a for loop
    num_controls = u_history.shape[1]
    fig, axs = plt.subplots(num_controls, 1, figsize=(10, 2 * num_controls))
    
    # Handle case where there's only one control dimension
    if num_controls == 1:
        axs = [axs]  # Make it iterable
        
    for i in range(num_controls):
        # Remove the [:-1] slice since time is already matched to u_history
        axs[i].plot(time, u_history[:, i], label=f'Control {i+1}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel(f'Control {i+1}')
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    plt.show()

    return fig, axs

def visualize_traj(s_history, save_path=None, show_plot=False):
    """
    Visualize a 2D slice (xy plane) of the value function for different dynamics models.
    
    Args:
        dyn_type: String indicating dynamics type ("DubinsSkater" or "SleighSkater")
        values: The computed value function array
        grid: The grid object
        times: Time array
        time_idx: Index of the time step to visualize
        theta, v, omega: Values for these dimensions (will find nearest on grid)
        title: Optional title for the plot
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        s_history: Optional trajectory data to overlay
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add labels and legend
    ax.set_title('Skater Trajectory')
    ax.set_xlabel('x position')
    ax.set_ylabel('y position')

    # Set equal axis
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, RINK_LENGTH)
    ax.set_ylim(0, RINK_WIDTH)
    
    # Add hockey rink elements
    for i in range(NUM_OPS):
        opponent = plt.Circle((OPPONENT_X[i], OPPONENT_Y[i]), OPPONENT_R, color='red', alpha=0.5, 
                         label='Opponent' if i == 0 else "")
        ax.add_patch(opponent)
    
    goal = plt.Circle((GOAL_X, GOAL_Y), GOAL_R, color='green', alpha=0.5, label='Goal Area')
    ax.add_patch(goal)
    
    rink = plt.Rectangle((0, 0), RINK_LENGTH, RINK_WIDTH, fill=False, edgecolor='blue', linestyle='--', label='Rink Boundary')
    ax.add_patch(rink)

    # Plot trajectory if provided
    if s_history is not None:
        plt.plot(s_history[:, 0], s_history[:, 1], 'o-', markersize=2, label='Skater Trajectory')

    ax.legend(loc='best')    
    
    # Save if requested
    if save_path is not None:
        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax

def animate_trajectory_with_value_function(s_history, t_history, values=None, grid=None, 
                                         times=None, dyn_type=None, theta=0, v=0, omega=0,
                                         show_value_function=True, title=None, 
                                         save_path=None, fps=10, get_opponent_position_fn=None):
    """
    Creates an animation of the skater's trajectory over time with optional value function background.
    The value function slice adapts to the current state of the skater.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for animation
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set axis limits and labels
    ax.set_xlim(0, RINK_LENGTH)
    ax.set_ylim(0, RINK_WIDTH)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal', adjustable='box')
    
    # Pre-compute opponent trajectories if function provided
    opponent_trajectories = None
    if get_opponent_position_fn and t_history is not None:
        opponent_trajectories = get_opponent_trajectory(get_opponent_position_fn, t_history)
    
    # Add hockey rink elements (store opponent patches in a list)
    opponents = []
    for i in range(NUM_OPS):
        # Use initial position at t=0
        if opponent_trajectories:
            ox, oy = opponent_trajectories[i][0]
        elif get_opponent_position_fn:
            ox, oy = get_opponent_position_fn(i, t_history[0])
        else:
            ox, oy = OPPONENT_X[i], OPPONENT_Y[i]
            
        opponent_patch = plt.Circle((ox, oy), OPPONENT_R, color='red', 
                                  alpha=0.5, label='Opponent' if i == 0 else "")
        opponents.append(opponent_patch)
        ax.add_patch(opponent_patch)
    
    goal = plt.Circle((GOAL_X, GOAL_Y), GOAL_R, color='green', alpha=0.5, label='Goal Area')
    ax.add_patch(goal)
    
    rink = plt.Rectangle((0, 0), RINK_LENGTH, RINK_WIDTH, fill=False, 
                        edgecolor='blue', linestyle='--', label='Rink Boundary')
    ax.add_patch(rink)
    
    # Initialize value function visualization if requested
    vbar = 3  # Value range for color scale
    mesh = None
    vf_time_text = None
    state_params_text = None
    
    if show_value_function and values is not None and grid is not None and times is not None:
        # Function to extract the appropriate 2D slice from value function
        # Updated to use current state values for theta and v
        def get_value_slice(time_idx, current_state=None):
            if dyn_type == "DubinsSkater":
                # Extract theta and v from current state if provided
                if current_state is not None:
                    current_theta = current_state[2]
                    current_v = current_state[3]
                else:
                    current_theta = theta
                    current_v = v
                
                # Find closest indices for current theta and v
                theta_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[2]) - current_theta))
                v_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[3]) - current_v))
                
                # Get the actual values from the grid
                theta_val = grid.coordinate_vectors[2][theta_idx]
                v_val = grid.coordinate_vectors[3][v_idx]
                
                # Extract slice for visualization
                value_slice = values[time_idx, :, :, theta_idx, v_idx]
                x_coords = grid.coordinate_vectors[0]
                y_coords = grid.coordinate_vectors[1]
                
                return value_slice, x_coords, y_coords, theta_val, v_val
                
            elif dyn_type == "SleighSkater":
                # Similar updates for SleighSkater
                if current_state is not None:
                    current_theta = current_state[2]
                    # For SleighSkater we might use different state components
                    # Assuming v_a and v_b are state components 7 and 8
                    current_v_a = current_state[7]
                    current_v_b = current_state[8]
                else:
                    current_theta = theta
                    current_v_a = v
                    current_v_b = omega
                
                theta_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[2]) - current_theta))
                a_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[5]) - 0))
                b_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[6]) - 0))
                v_a_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[7]) - current_v_a))
                v_b_idx = np.argmin(np.abs(np.array(grid.coordinate_vectors[8]) - current_v_b))
                
                value_slice = values[time_idx, 2, 2, theta_idx, :, :, a_idx, b_idx, v_a_idx, v_b_idx]
                x_coords = grid.coordinate_vectors[3]
                y_coords = grid.coordinate_vectors[4]
                
                return value_slice, x_coords, y_coords, current_theta, current_v_a
            else:
                raise ValueError(f"Unknown dynamics type: {dyn_type}")
        
        # Initialize with final time slice (using default values)
        value_slice, x_coords, y_coords, _, _ = get_value_slice(-1)
        
        # Create mesh for value function
        mesh = ax.pcolormesh(
            x_coords, y_coords, value_slice.T,
            cmap='RdBu_r', vmin=-vbar, vmax=vbar, alpha=0.7
        )
        fig.colorbar(mesh, ax=ax, label='Value Function')
        
        # Add text displays with better positioning and background properties
        text_bg_props = dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5')
        
        # Main time text at top left
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          fontsize=12, color='black',
                          bbox=text_bg_props)
        
        # Value function time below with more space
        vf_time_text = ax.text(0.02, 0.92, f'Value Function Time: {times[-1]:.2f}s',
                            transform=ax.transAxes, fontsize=12, color='black',
                            bbox=text_bg_props)
        
        # State parameters even lower
        state_params_text = ax.text(0.02, 0.86, '',
                            transform=ax.transAxes, fontsize=12, color='black',
                            bbox=text_bg_props)
    else:
        # If no value function, just add time text at the top with good background
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                          fontsize=12, color='black',
                          bbox=dict(facecolor='white', alpha=0.7, 
                                   edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # Initialize dynamic elements
    trajectory_line, = ax.plot([], [], 'b-', lw=2, label='Path')
    skater_point, = ax.plot([], [], 'bo', ms=10, label='Skater')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Skater Trajectory')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Calculate number of frames
    n_frames = min(100, len(s_history))
    frame_indices = np.linspace(0, len(s_history)-1, n_frames, dtype=int)
    
    # Initialize function for animation
    def init():
        trajectory_line.set_data([], [])
        skater_point.set_data([], [])
        time_text.set_text('')
        if state_params_text is not None:
            state_params_text.set_text('')
        return [trajectory_line, skater_point, time_text]
    
    # Update function for animation
    def update(frame_idx):
        i = frame_indices[frame_idx]
        current_state = s_history[i]
        current_time = t_history[i]
        
        # Update trajectory and skater position
        trajectory_line.set_data(s_history[:i+1, 0], s_history[:i+1, 1])
        skater_point.set_data([current_state[0]], [current_state[1]])
        time_text.set_text(f'Time: {current_time:.2f}s')
        
        # Update opponent positions
        if opponent_trajectories:
            for j, opponent in enumerate(opponents):
                opponent.center = opponent_trajectories[j][i]
        elif get_opponent_position_fn:
            for j, opponent in enumerate(opponents):
                ox, oy = get_opponent_position_fn(j, current_time)
                opponent.center = (ox, oy)
        
        # Update value function if shown
        if show_value_function and mesh is not None and values is not None:
            # Calculate value function time index based on animation progress
            progress_ratio = i / (len(s_history) - 1)
            value_time_idx = max(0, min(len(values)-1, 
                                    int(len(values) - 1 - progress_ratio * (len(values) - 1))))
            
            # Update mesh data with current state
            value_slice, x_coords, y_coords, theta_val, v_val = get_value_slice(value_time_idx, current_state)
            mesh.set_array(value_slice.T.flatten())
            
            # Update time and state parameter text
            if vf_time_text is not None:
                vf_time_text.set_text(f'Value Function Time: {times[value_time_idx]:.2f}s')
            
            if state_params_text is not None:
                if dyn_type == "DubinsSkater":
                    state_params_text.set_text(f'θ: {theta_val:.2f}, v: {v_val:.2f}')
                elif dyn_type == "SleighSkater":
                    state_params_text.set_text(f'θ: {theta_val:.2f}, v_a: {v_val:.2f}')
        
        # Return updated elements
        return [trajectory_line, skater_point, time_text]
    
    # Create animation
    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        blit=(state_params_text is None),  # Only use blit=True if we're not updating text
        interval=1000/fps
    )
    
    # Save animation with robust approach
    if save_path:
        if not os.path.isabs(save_path):
            save_path = os.path.join(OUTPUT_DIR, save_path)
        
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        print(f"Saving animation to: {save_path}")
        try:
            # First try matplotlib's default writer
            anim.save(save_path, fps=fps, dpi=100)
            print(f"Animation saved successfully!")
        except Exception as e1:
            print(f"Default save failed: {e1}")
            try:
                # Try with specific writer based on file extension
                if save_path.lower().endswith('.mp4'):
                    # Try FFmpeg
                    writer = 'ffmpeg'
                    anim.save(save_path, writer=writer, fps=fps, dpi=100)
                elif save_path.lower().endswith('.gif'):
                    # Try pillow
                    writer = 'pillow'
                    anim.save(save_path, writer=writer, fps=fps, dpi=100)
                else:
                    # Default for other formats
                    anim.save(save_path, fps=fps, dpi=100)
                print(f"Animation saved successfully with alternate writer!")
            except Exception as e2:
                print(f"All save attempts failed. Error: {e2}")
                
                # Last resort: save as PNG sequence
                try:
                    base_path = os.path.splitext(save_path)[0]
                    print(f"Saving individual frames to {base_path}_frames/")
                    os.makedirs(f"{base_path}_frames", exist_ok=True)
                    
                    for i in range(n_frames):
                        update(i)
                        plt.savefig(f"{base_path}_frames/frame_{i:03d}.png", dpi=100)
                    
                    print("Saved individual frames. You can combine them with:")
                    print(f"ffmpeg -framerate {fps} -i {base_path}_frames/frame_%03d.png -c:v libx264 {save_path}")
                except Exception as e3:
                    print(f"Failed to save frames: {e3}")
    
    plt.close(fig)
    return anim
