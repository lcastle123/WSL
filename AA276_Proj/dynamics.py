import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as  hj

from hj_reachability import dynamics
from hj_reachability import sets
from constants import *

class DubinsSkater(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, v_dot_bar=10., omega_bar=4.,
                 control_mode="min", disturbance_mode="max",
                 control_space=None, disturbance_space=None):
        self.v_dot_bar = v_dot_bar
        self.omega_bar = omega_bar
        
        # 4D grid: [x, y, theta, v]
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
                hj.sets.Box(np.array([0, 0, -2*jnp.pi, 0]), 
                            np.array([RINK_LENGTH, RINK_WIDTH, 2*jnp.pi, 10])),
                (121, 121, 21, 21)) 

        if control_space is None:
            # Controls are [v_cmd, omega]
            control_space = hj.sets.Box(jnp.array([-v_dot_bar, -omega_bar]), jnp.array([v_dot_bar, omega_bar]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([0., 0.]), jnp.array([0., 0.]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, theta, v = state
        return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), 0, 0])  # Natural dynamics with current velocity

    def control_jacobian(self, state, time):
        x, y, theta, v = state
        return jnp.array([
            [0., 0.],          # v_cmd doesn't directly affect x
            [0., 0.],          # v_cmd doesn't directly affect y
            [0., 1.],          # omega directly affects theta
            [1., 0.]           # v_cmd directly affects v
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ])

    def f_cl(self, s, u):
        """Compute the dynamics of the skater with v_cmd and omega as controls"""
        v_cmd, omega = u
        x, y, theta, v = s
        
        # Simple model: v approaches v_cmd with some time constant
        v_dot = (v_cmd - v)  # v_cmd directly affects v with time constant 1
        
        return jnp.array([
            v * jnp.cos(theta),
            v * jnp.sin(theta),
            omega,
            v_dot
        ])


class SleighSkater(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, M=70, m=10, I=10, u_1_bar = 2., u_2_bar = 2.,
                 control_mode="max", disturbance_mode="min",
                 control_space=None, disturbance_space=None):
        self.u_1_bar = u_1_bar
        self.u_2_bar = u_2_bar
        self.M = M
        self.m = m
        self.I = I

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
                hj.sets.Box(np.array([0, 0, -jnp.pi, -10, -5, -1, -1, -5, -5]), 
                            np.array([RINK_LENGTH, RINK_WIDTH, jnp.pi, 10, 5, 1, 1, 5, 5])),
                (11,11,4,4,4,4,4,4,4))

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-u_1_bar, -u_2_bar]), jnp.array([u_1_bar, u_2_bar]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([0., 0.]), jnp.array([0., 0.]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, theta, p1, p2, a, b, v_a, v_b = state

        M = self.M
        m = self.m
        I = self.I

        denom = (M + m) * (I + m * a**2) + M * m * b**2

        xi_1_numerator = (M + m) * (p1 - m * a * v_b) + m * b * (p2 + M * v_a)
        xi_1 = xi_1_numerator / denom

        xi_2_numerator = (
            -m * (b * (p1 - m * a * v_b) - (I + m * a**2) * v_a)
            + (I + m * (a**2 + b**2)) * p2
        )
        xi_2 = xi_2_numerator / denom

        eta_numerator = (
            (M * m * b**2 + I * (M + m)) * b
            + a * ((M + m) * p1 + m * b * (p2 + M * v_a))
        )
        eta = eta_numerator / denom
        
        return jnp.array([xi_2 * jnp.cos(theta),
                          -xi_2 * jnp.sin(theta),
                          xi_1,
                          -m*eta*xi_2,
                          m*eta*xi_1,
                          v_a,
                          v_b,
                          0,
                          0])

    def control_jacobian(self, state, time):
        x, y, theta, p1, p2, a, b, v_a, v_b = state
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ])
    

    def f_cl(self, s, u):
        '''
        Compute the dynamics of the skater. 
        '''
        return self.open_loop_dynamics(s, 0) + self.control_jacobian(s, 0) @ u

