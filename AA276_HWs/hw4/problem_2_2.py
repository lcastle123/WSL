import numpy as np
from problem_1_2 import CartPole, save_values_gif
import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as  hj
import cvxpy as cp

from hj_reachability import dynamics
from hj_reachability import sets
from scipy.interpolate import interpn

import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class Controller:
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
        self.u_nom = 10.
        self.d_bar = 5.
        self.n = 2
        self.m_p = 1
        self.m_c = 1
        self.l = 1
        self.g = 2
        self.dt = 0.01
        self.start = True

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(np.array([0, -10]), np.array([2*np.pi, 10])),(101, 101))
        self.failure_values = -jnp.abs((self.grid.states[...,0]%(2*jnp.pi)) - jnp.pi) + jnp.pi/2
        self.times = np.linspace(0, -10, 101, endpoint=True)
        self.dynamics = CartPole(d_bar_low=-self.d_bar, d_bar_up=self.d_bar)
        self.solver_settings = hj.SolverSettings.with_accuracy("very_high", 
                                                        hamiltonian_postprocessor=
                                                        hj.solver.backwards_reachable_tube)
        values = hj.solve(self.solver_settings,self.dynamics, self.grid, self.times, self.failure_values)
        self.values = values[-1]

    def reset(self):
        self.s_history = []
        self.t_history = []
        self.u_history = []
        self.d_estimate_history = []

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

        if len(self.d_estimate_history) % 100 == 0 and len(self.d_estimate_history) > 0:
            d_bar_low = np.min(self.d_estimate_history[-5:]) - 1.5
            d_bar_up = np.max(self.d_estimate_history[-5:]) + 1.5
            self.dynamics = CartPole(d_bar_low=d_bar_low, d_bar_up=d_bar_up)
            values = hj.solve(self.solver_settings,self.dynamics, self.grid, self.times, self.failure_values)
            self.values = values[-1]

        if t == 0:
            d_estimate = np.random.uniform(-5, 5)
            self.d_estimate_history.append(d_estimate)
        else:
            theta_dot_prev = self.s_history[-2][3]
            theta_dot = s[3]
            d_theta_dot = (theta_dot - theta_dot_prev)/ (t - self.t_history[-2])
            u_prev = self.u_history[-1]
            state = jnp.array([s[1], s[3]])
            f_val = self.dynamics.open_loop_dynamics(state,None)[-1]
            g_val = self.dynamics.control_jacobian(state,None)
            g_val = g_val * u_prev
            g_val = g_val[-1]

            d_estimate = d_theta_dot - f_val - g_val
            self.d_estimate_history.append(d_estimate.item())

        if self.start:
            u = self.lqr_control(s, t)
            if s[1] < jnp.pi/2*1.1:
                self.start = False
        else: 
            u = self.smooth_blending_safety_filter(s, self.u_nom, self.d_estimate_history[-1], 15, 1e4)
        
        self.u_history.append(u)

        if len(self.u_history) == 995:
            self.plot_trajectory()
        #print("theta", s[1])
        return np.array([u])    

    def data_to_visualize(self):
        """
        Use this to add any number of data visualizations to the animation.
        This is purely to help you debug, in case you find it helpful.
        See example code below to plot the control on a new axes at axes index 2
        and the disturbance estimate on an existing axes at axes index 1.

        Returns:
            data_to_visualize (dict): Each dictionary entry should have the form:
                'y-axes label' (str): [axes index (int), data to visualize (np.ndarray), line styles (dict)]
        """
        s_history = np.array(self.s_history)
        t_history = np.array(self.t_history)
        u_history = np.array(self.u_history)
        d_estimate_history = np.array(self.d_estimate_history)
        return {
            'u (N)': [2, u_history, {'color': 'k'}],
            '$\\hat{d}$ (rad/s$^2$)': [1, d_estimate_history, {'color': 'k', 'linestyle': '--'}],
            '$\\theta$ (rad)': [3, s_history[:, 1] % (2*np.pi), {'color': 'k'}],
            '$\\theta_\\text{min}$ (rad)': [3, (np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}],
            '$\\theta_\\text{max}$ (rad)': [3, (3*np.pi/2)*np.ones(len(t_history)), {'color': 'r', 'linestyle': '--'}]
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

    def plot_trajectory(self, save_path='outputs/trajectory.png'):
        plt1, ax1 = plt.subplots()
        ax1.plot(np.array(self.s_history)[:, 1], np.array(self.s_history)[:, 3], linewidth=2, color='purple')
        plt.title(f'Optimally Safe Trajectory')
        plt.xlabel('$theta$ (rad)')
        plt.ylabel('$theta dot$ (rad/s)')
        self.plot_value_and_safe_set_boundary(ax1)
        ax1.contour(
            self.grid.coordinate_vectors[0],
            self.grid.coordinate_vectors[1],
            self.failure_values.T,
            levels=0,
            colors='r'
        )
        plt.xlim(0, 2*np.pi)
        plt.ylim(-10., 10.)
        plt1.savefig(save_path,            # or .pdf, .svg, .jpg, etc.
                    dpi=300,                   # resolution in dots per inch
                    bbox_inches="tight")       # crop to the content
    
    def plot_value_and_safe_set_boundary(self, ax):
        """
        args:
            values_converged: ndarray with shape [
                    len(grid.coordinate_vectors[0]),
                    len(grid.coordinate_vectors[1])
                ]
            grid: hj.Grid,
            ax: matplotlib axes to plot on
        """
        values_converged_interpolator = RegularGridInterpolator(
            ([np.array(v) for v in self.grid.coordinate_vectors]),
            np.array(self.values),
            bounds_error=False,
            fill_value=None
        )
        vbar=3
        vis_thetas = np.linspace(0, np.pi*2, num=101, endpoint=True)
        vis_theta_dots = np.linspace(-10, +10, num=101, endpoint=True)
        vis_xs = np.stack((np.meshgrid(vis_thetas, vis_theta_dots, indexing='ij')), axis=2)
        vis_values_converged = values_converged_interpolator(vis_xs)
        ax.pcolormesh(
            vis_thetas,
            vis_theta_dots,
            vis_values_converged.T,
            cmap='RdBu',
            vmin=-vbar,
            vmax=vbar
        )
        ax.contour(
            vis_thetas,
            vis_theta_dots,
            vis_values_converged.T,
            levels=[0],
            colors='k'
        )

    def lqr_control(self, s, t):
        """
        Linear Quadratic Regulator (LQR) control for the cart-pole system.
        This is a placeholder function and should be implemented as needed.

        Args:
            s (np.ndarray): The current state: [x, theta, x_dot, theta_dot]
            t (float): The current time

        Returns:
            u (np.ndarray): The control input [u]
        """
        s = jnp.array(jnp.array([s[1], s[3]], dtype=jnp.float32)) 
        dt = self.dt
        Q = jnp.eye(self.n)  # State cost matrix
        R = jnp.array([[1]])
        A = np.array([
            [0, 1],
            [(self.m_c+self.m_p)*self.g/(self.l*self.m_c), 0]
            ])
        B = np.array([
            [0],
            [1 / (self.l * (self.m_c))]
        ])
        A = np.eye(self.n) + dt * A
        B = dt * B

        eps = 1e-4  # Riccati recursion convergence tolerance
        max_iters =2000  # Riccati recursion maximum number of iterations
        P_prev = np.zeros((self.n, self.n))  # initialization
        converged = False
        for _ in range(max_iters):
            # Apply the Ricatti equation until convergence
            K = -np.linalg.inv(R + B.T @ P_prev @ B) @ (B.T @ P_prev @ A)
            P = Q + A.T @ P_prev@ (A + B @ K)
            if np.max(np.abs(P - P_prev)) < eps:
                converged = True
                break
            P_prev = P
        if not converged:
            raise RuntimeError("Ricatti recursion did not converge!")
        #print("K:", K)

        s_tilde = s - np.array([jnp.pi/2*1.1 , 0])
        u = K @ s_tilde

        return u.item()
