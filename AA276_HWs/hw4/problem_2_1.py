import numpy as np
from problem_1_2 import CartPole
import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as  hj
import cvxpy as cp

from hj_reachability import dynamics
from hj_reachability import sets
from scipy.interpolate import interpn




class ControllerA:
    """
    Controller for the cart-pole system.
    This is a template for you to implement however you desire.
    
    reset(.) is called before each cart-pole simulation.
    u_fn(.) is called at each simulation step.
    data_to_visualize(.) is called after each simulation.

    We provide example code for a random-walk controller.
    """

    def __init__(self):
        self.reset()

        self.u_nom = -10
        self.d = -1.

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([0, -10]), np.array([2*np.pi, 10])),(101, 101))
        failure_values = -jnp.abs((self.grid.states[...,0]%(2*jnp.pi)) - jnp.pi) + jnp.pi/2
        times = np.linspace(0, -10, 101, endpoint=True)
        self.dynamics = CartPole(d_bar=self.d)
        self.solver_settings = hj.SolverSettings.with_accuracy("very_high", 
                                                        hamiltonian_postprocessor=
                                                        hj.solver.backwards_reachable_tube)
        values = hj.solve(self.solver_settings,self.dynamics, self.grid, times, failure_values)
        self.values = values[-1]
        

    def reset(self):
        self.s_history = []
        self.t_history = []
        self.u_history = []


    def u_fn(self, s, t):
        """Control function for the cart-pole system.

        Args:
            s (np.ndarray): The current state: [x, theta, x_dot, theta_dot]
                NOTE: you might want to first wrap theta in [0, 2pi]
            t (float): The current time

        Returns:
            u (np.ndarray): The control input [u]
        """
        self.s_history.append(s)
        self.t_history.append(t)
        filter_type = 3
        value = self.query_value_function(jnp.array([s[1],s[3]]))
        if value > 0.5 and filter_type != 3:
            u = self.u_nom
        else:
            if filter_type == 1:
                u = self.least_restricive_filter(s)
                u = u * self.u_nom
            elif filter_type == 2:
                #u = self.smooth_blending_safety_filter(s, self.u_nom, 10, 1e4)
                pass
            elif filter_type == 3:
                u = self.smooth_blending_safety_filter(s, self.u_nom, self.d, 5, 1e3)
        self.u_history.append(u)      
        return np.array([u])                      

    def data_to_visualize(self):
        """
        Use this to add any number of data visualizations to the animation.
        This is purely to help you debug, in case you find it helpful.
        See example code below to plot the control on a new axes at axes index 1.

        Returns:
            data_to_visualize (dict): Each dictionary entry should have the form:
                'y-axes label' (str): [axes index (int), data to visualize (np.ndarray), line styles (dict)]
        """
        s_history = np.array(self.s_history)
        t_history = np.array(self.t_history)
        u_history = np.array(self.u_history)
        return {
            'u (N)': [1, u_history, {'color': 'k'}],
            '$\\theta$ (rad)': [2, s_history[:, 1] % (2*np.pi), {'color': 'k'}],
            '$\\theta_\\text{min}$ (rad)': [2, (np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}],
            '$\\theta_\\text{max}$ (rad)': [2, (3*np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}]
        }
    
    def query_value_function(self, s):
        values_converged = self.values
        v = interpn(
        ([np.array(v) for v in self.grid.coordinate_vectors]),  # grid vectors
        np.array(values_converged),                       # level-set array
        np.atleast_2d(s),                                 # shape (1, 2)
        method='linear',
        bounds_error=False,
        fill_value=None
        )
        # Wrap theta in [0, 2pi]
        x = s[1] % (2 * np.pi)
        return v.item()
    
    def grad_at_state(self, s):
        grads = self.grid.grad_values(self.values, self.solver_settings.upwind_scheme)
        return np.array([
            interpn(
                points=[np.array(v) for v in self.grid.coordinate_vectors],
                values=np.array(grads[:,:,i]),
                xi=np.array(s),
                method='linear',
                bounds_error=False,
                fill_value=0.0
            ).item()
            for i in range(2)
        ])

    def smooth_blending_safety_filter(self, state, u_nom, d, gamma, lmbda):
        
        state = jnp.array([state[1],state[3]])
        u_sb = cp.Variable(1)
        s = cp.Variable(1)
        u_bar = 10
        u_upper = u_bar
        u_lower = -u_bar

        obj = cp.Minimize(cp.sum_squares(u_sb - u_nom) + lmbda * cp.square(s))
        constr = [u_sb >= u_lower, u_sb <= u_upper]

        f_val = self.dynamics.open_loop_dynamics(state,None)
        g_val = self.dynamics.control_jacobian(state,None)
        d_val = self.dynamics.disturbance_jacobian(state,None)
        dyn = f_val + g_val @ u_sb + d_val * d

        v, grad_v = (self.query_value_function(state), self.grad_at_state(state))
        
        constr += [cp.matmul(grad_v, dyn) + gamma * v + s >= 0]
        constr += [s >= 0]
        prob = cp.Problem(obj, constr)
        try:
            prob.solve()
        except cp.error.SolverError as e:
            print(f"Solver error: {e}")
            return self.dynamics.optimal_control((state), 0.0, grad_v).item()
        return u_sb.value[0] if u_sb.value is not None else self.dynamics.optimal_control((state), 0.0, grad_v).item()
    
    def least_restricive_filter(self, state):
        grads = self.grid.grad_values(self.values, self.solver_settings.upwind_scheme)
        beta2s = grads[:, :, 1]
        beta2 = interpn(
            ([np.array(v) for v in self.grid.coordinate_vectors]),
            np.array(beta2s),
            state,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        return np.sign(beta2[1]).item()
    
class ControllerB:
    """
    Controller for the cart-pole system.
    This is a template for you to implement however you desire.
    
    reset(.) is called before each cart-pole simulation.
    u_fn(.) is called at each simulation step.
    data_to_visualize(.) is called after each simulation.

    We provide example code for a random controller.
    """

    def __init__(self):
        self.reset()

        self.u_nom = -10
        self.d = 1.

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([0, -10]), np.array([2*np.pi, 10])),(101, 101))
        failure_values = -jnp.abs((self.grid.states[...,0]%(2*jnp.pi)) - jnp.pi) + jnp.pi/2
        times = np.linspace(0, -10, 101, endpoint=True)
        self.dynamics = CartPole(d_bar=self.d)
        self.solver_settings = hj.SolverSettings.with_accuracy("very_high", 
                                                        hamiltonian_postprocessor=
                                                        hj.solver.backwards_reachable_tube)
        values = hj.solve(self.solver_settings,self.dynamics, self.grid, times, failure_values)
        self.values = values[-1]
        

    def reset(self):
        self.s_history = []
        self.t_history = []
        self.u_history = []


    def u_fn(self, s, t):
        """Control function for the cart-pole system.

        Args:
            s (np.ndarray): The current state: [x, theta, x_dot, theta_dot]
                NOTE: you might want to first wrap theta in [0, 2pi]
            t (float): The current time

        Returns:
            u (np.ndarray): The control input [u]
        """
        self.s_history.append(s)
        self.t_history.append(t)
        filter_type = 3
        value = self.query_value_function(jnp.array([s[1],s[3]]))
        if value > 0.5 and filter_type != 3:
            u = self.u_nom
        else:
            if filter_type == 1:
                u = self.least_restricive_filter(s)
                u = u * self.u_nom
            elif filter_type == 2:
                #u = self.smooth_blending_safety_filter(s, self.u_nom, 10, 1e4)
                pass
            elif filter_type == 3:
                u = self.smooth_blending_safety_filter(s, self.u_nom, self.d, 5, 1e3)
        self.u_history.append(u)      
        return np.array([u])                      

    def data_to_visualize(self):
        """
        Use this to add any number of data visualizations to the animation.
        This is purely to help you debug, in case you find it helpful.
        See example code below to plot the control on a new axes at axes index 1.

        Returns:
            data_to_visualize (dict): Each dictionary entry should have the form:
                'y-axes label' (str): [axes index (int), data to visualize (np.ndarray), line styles (dict)]
        """
        s_history = np.array(self.s_history)
        t_history = np.array(self.t_history)
        u_history = np.array(self.u_history)
        return {
            'u (N)': [1, u_history, {'color': 'k'}],
            '$\\theta$ (rad)': [2, s_history[:, 1] % (2*np.pi), {'color': 'k'}],
            '$\\theta_\\text{min}$ (rad)': [2, (np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}],
            '$\\theta_\\text{max}$ (rad)': [2, (3*np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}]
        }
    
    def query_value_function(self, s):
        values_converged = self.values
        v = interpn(
        ([np.array(v) for v in self.grid.coordinate_vectors]),  # grid vectors
        np.array(values_converged),                       # level-set array
        np.atleast_2d(s),                                 # shape (1, 2)
        method='linear',
        bounds_error=False,
        fill_value=None
        )
        # Wrap theta in [0, 2pi]
        x = s[1] % (2 * np.pi)
        return v.item()
    
    def grad_at_state(self, s):
        grads = self.grid.grad_values(self.values, self.solver_settings.upwind_scheme)
        return np.array([
            interpn(
                points=[np.array(v) for v in self.grid.coordinate_vectors],
                values=np.array(grads[:,:,i]),
                xi=np.array(s),
                method='linear',
                bounds_error=False,
                fill_value=0.0
            ).item()
            for i in range(2)
        ])

    def smooth_blending_safety_filter(self, state, u_nom, d, gamma, lmbda):
        
        state = jnp.array([state[1],state[3]])
        u_sb = cp.Variable(1)
        s = cp.Variable(1)
        u_bar = 10
        u_upper = u_bar
        u_lower = -u_bar

        obj = cp.Minimize(cp.sum_squares(u_sb - u_nom) + lmbda * cp.square(s))
        constr = [u_sb >= u_lower, u_sb <= u_upper]

        f_val = self.dynamics.open_loop_dynamics(state,None)
        g_val = self.dynamics.control_jacobian(state,None)
        d_val = self.dynamics.disturbance_jacobian(state,None)
        dyn = f_val + g_val @ u_sb + d_val * d

        v, grad_v = (self.query_value_function(state), self.grad_at_state(state))
        
        constr += [cp.matmul(grad_v, dyn) + gamma * v + s >= 0]
        constr += [s >= 0]
        prob = cp.Problem(obj, constr)
        try:
            prob.solve()
        except cp.error.SolverError as e:
            print(f"Solver error: {e}")
            return self.dynamics.optimal_control((state), 0.0, grad_v).item()
        return u_sb.value[0] if u_sb.value is not None else self.dynamics.optimal_control((state), 0.0, grad_v).item()
    
    def least_restricive_filter(self, state):
        grads = self.grid.grad_values(self.values, self.solver_settings.upwind_scheme)
        beta2s = grads[:, :, 1]
        beta2 = interpn(
            ([np.array(v) for v in self.grid.coordinate_vectors]),
            np.array(beta2s),
            state,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        return np.sign(beta2[1]).item()
