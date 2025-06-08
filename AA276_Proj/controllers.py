import numpy as np
import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as  hj
import cvxpy as cp

from scipy.interpolate import interpn
from constants import *
from visualizations import visualize_xy_slice

class Controller:
    """
    
    """

    def __init__(self, values, dynamics, grid, solver_settings=None, controller_type='LeastRestrictive', times = None):
        self.controller_type = controller_type
        self.values = values
        self.values_converged = values[-1]  # Assuming the last time step is the converged values
        self.dynamics = dynamics
        self.grid = grid
        self.times = times if times is not None else np.linspace(0, T, VALS_NT)
        if solver_settings is None:
            self.solver_settings = hj.SolverSettings.with_accuracy("very_high", 
                                                        hamiltonian_postprocessor=
                                                        hj.solver.backwards_reachable_tube)
        else:
            self.solver_settings = solver_settings
        self.grads_converged = self.grid.grad_values(self.values_converged, self.solver_settings.upwind_scheme)
        # Initialize grads with the converged gradients
        self.grads = self.grads_converged
        self.vals_idx = 0
    
    def controller(self, s, t):
        if self.controller_type == 'SafeController':
            return self.safe_constroller(s,t)
        elif self.controller_type == 'LeastRestrictive':
            return self.least_restrictive_controller(s, t)
        elif self.controller_type == 'Nominal':
            return self.nominal_controller(s, t)


    def value_at_state(self, s, t=None):
        if t is not None:
            vals_dt = T / (VALS_NT-1)
            vals_idx = -int(jnp.ceil(t/vals_dt))
            values_at_t = self.values[vals_idx]
            value = self.grid.interpolate(values_at_t, s)
        else:
            value = self.grid.interpolate(self.values_converged, s)
        return value
    
    def grad_value_at_state(self, s, t=None, visualize=False):
        if t is not None:
            vals_dt = T / (VALS_NT-1)
            vals_idx = -int(jnp.ceil(t/vals_dt))
            if self.vals_idx != vals_idx:
                values_at_t = self.values[vals_idx]
                self.grads = self.grid.grad_values(values_at_t, self.solver_settings.upwind_scheme)
                if visualize:
                    visualize_xy_slice("DubinsSkater", self.values, self.grid, self.times, vals_idx,
                                theta=s[2], save_path=f'brt_t{vals_idx}.png', s_history=s)
                self.vals_idx = vals_idx
            grads = self.grid.interpolate(self.grads, s)
        else:
            grads = self.grid.interpolate(self.grads_converged, s)
        return grads

    def safe_constroller(self, s, t):
        """
        Computes the safest control action at state s and time t.
        """
        # Get the gradient of the value function at state s
        grad_value = self.grad_value_at_state(s, t)
        #print(f"Gradient at state {s} at time {t}: {grad_value}")
        u = self.dynamics.optimal_control((s), t, grad_value)
        return u
    
    def least_restrictive_controller(self, s, t):
        """
        Computes the least restrictive control action at state s and time t.
        """
        u = self.nominal_controller(s, t)
        val_at_s = self.value_at_state(s, t)
        
        if val_at_s > -1:
            # If the value function is positive, we are in a bad state
            u = self.safe_constroller(s, t)
        
        return u
    
    def nominal_controller(self, s, t):
        """Modified for v, omega control with 4D state"""
        # Calculate heading to goal
        theta_target = np.arctan2(GOAL_Y - s[1], GOAL_X - s[0])
        theta_diff = theta_target - s[2]
        # Normalize theta_diff to be within [-pi, pi]
        theta_diff = (theta_diff + np.pi) % (2 * np.pi) - np.pi
        
        # Control to steer towards the goal - direct omega control
        omega = np.clip(theta_diff * 2.0, -self.dynamics.omega_bar, self.dynamics.omega_bar)
        
        # Speed control - target speed based on distance
        distance_to_goal = np.linalg.norm([GOAL_X - s[0], GOAL_Y - s[1]])
        v_target = np.clip(distance_to_goal, 1.0, self.dynamics.v_dot_bar)
        
        # If current speed is too high, slow down; if too low, speed up
        v_cmd = np.clip(v_target, 0, self.dynamics.v_dot_bar)
        
        return np.array([v_cmd, omega])
