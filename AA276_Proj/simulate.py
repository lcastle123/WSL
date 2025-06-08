import numpy as np
from tqdm import tqdm
from constants import N, M
from controllers import Controller
from constants import *

def simulate(s0, controller:Controller, T, f, dt=DT, get_opponent_position_fn=None):
    """
    Simulate the system dynamics forward in time.
    
    Args:
        s0: Initial state
        controller: Controller object
        T: Final time
        f: System dynamics function
        dt: Time step
        get_opponent_position_fn: Function to get opponent position at a given time
    """

    nt = int(T / dt)  # Number of time steps
    
    # Initialize arrays with proper dimensions
    s_history = np.full((nt+1, len(s0)), np.nan)  # +1 for initial state
    u_history = np.full((nt, M), np.nan)          # Control inputs (one fewer than states)
    t_history = np.full((nt+1), np.nan)           # Time points (matches states)
    
    # Set initial state and time
    s_history[0] = s0
    t_history[0] = 0.0
    time_to_goal = None

    for i in tqdm(range(nt), desc='Simulating...'):
        # Get control input for current state
        u = controller.controller(s_history[i, :], i*dt)
        u_history[i] = u
        t_history[i+1] = (i+1)*dt
        
        # Compute next state
        s_history[i+1] = s_history[i] + dt*f(s_history[i], u)
        
        # Clip state variables to stay within bounds
        s_history[i+1, 0] = np.clip(s_history[i+1, 0], 0, RINK_LENGTH)
        s_history[i+1, 1] = np.clip(s_history[i+1, 1], 0, RINK_WIDTH)
        s_history[i+1, 2] = np.mod(s_history[i+1, 2] + 2*np.pi, 4*np.pi) - 2*np.pi  # Keep theta in [-2π, 2π]
        s_history[i+1, 3] = np.clip(s_history[i+1, 3], 0, 10.)  # Keep velocity in valid range

        
        # Check terminal conditions
        goal_check = check_goal(s_history[i+1, :])
        current_time = (i+1)*dt
        fail_check = check_fail(s_history[i+1, :], current_time, get_opponent_position_fn)
        
        if np.any(goal_check):
            print(f"Goal reached at time {(i+1)*dt:.2f} seconds.")
            # After reaching the goal, hold state, set control to zero, and continue time
            s_history[i+2:] = s_history[i+1]  # Hold the final state
            u_history[i+1:] = 0.0             # Set remaining controls to zero
            t_history[i+2:] = np.arange(i+2, nt+1) * dt  # Continue time vector
            time_to_goal = (i+1) * dt
            break
        elif np.any(fail_check):
            print(f"Failure condition met at time {(i+1)*dt:.2f} seconds.")
            # Clip arrays to current step
            s_history[i+2:] = s_history[i+1]  # Hold the final state
            u_history[i+1:] = 0.0             # Set remaining controls to zero
            t_history[i+2:] = np.arange(i+2, nt+1) * dt  # Continue time vector
            time_to_goal = None  # No valid time to goal if failed
            break
        
        #if i % 50 == 0:
            #print(f"Step {i}, State: {s_history[i]}, Control: {u}")

    return s_history, u_history, t_history, time_to_goal

def check_goal(state):
    """
    Check if the skater has reached the goal.
    """
    x, y = state[:2]
    distance_to_goal = np.sqrt((x - GOAL_X)**2 + (y - GOAL_Y)**2)
    return distance_to_goal < GOAL_R

def check_fail(state, time, get_opponent_position_fn=None):
    """
    Check if the skater has failed by colliding with the opponent.
    """
    x, y = state[:2]
    
    distance_to_opponent = []
    for i in range(NUM_OPS):
        # Use the position function if provided, otherwise use static positions
        op_yis = np.array([OPPONENT_Y[j] for j in range(NUM_OPS)])
        if get_opponent_position_fn:
            ox, oy = get_opponent_position_fn(i, time)
            ox = ox + 10
            oy = oy + 5 * jnp.sign(op_yis[i] - RINK_WIDTH / 2)  # Adjust y position based on rink center
        else:
            ox, oy = OPPONENT_X[i], OPPONENT_Y[i]
        
        distance = np.sqrt((x - ox)**2 + (y - oy)**2)
        distance_to_opponent.append(distance)
    
    distance_to_opponent = np.array(distance_to_opponent)
    # Check if any state is nan
    if np.isnan(distance_to_opponent).any():
        return True
    return distance_to_opponent < OPPONENT_R
