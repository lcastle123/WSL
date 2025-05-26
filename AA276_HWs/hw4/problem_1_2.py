import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as  hj

from hj_reachability import dynamics
from hj_reachability import sets

class CartPole(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, m_p=1., m_c=1., l=1., g=2, u_bar=10., d_bar_up=0., d_bar_low=0.,
                 control_mode="max", disturbance_mode="min",
                 control_space=None, disturbance_space=None):
        self.m_p = m_p
        self.m_c = m_c
        self.l = l
        self.g = g
        #self.u_bar = u_bar_up
        #self.d = d_bar
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-u_bar]), jnp.array([u_bar]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([d_bar_low]), jnp.array([d_bar_up]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        theta, theta_dot = state
        return jnp.array([theta_dot,
                          -((self.m_c+self.m_p)*self.g*jnp.sin(theta)+self.m_p*self.l*theta_dot*theta_dot*jnp.sin(theta)*jnp.cos(theta))/(self.l*(self.m_c+self.m_p*jnp.sin(theta)*jnp.sin(theta)))])

    def control_jacobian(self, state, time):
        theta, _ = state
        return jnp.array([
            [0.],
            [ -jnp.cos(theta) / (self.l*(self.m_c+self.m_p*jnp.sin(theta)*jnp.sin(theta)))]
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [1.]
        ])
    
'''
dynamics = CartPole()
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(np.array([0, -10]), 
                np.array([2*np.pi, 10])),
    (101, 101))

# Define the failure set
failure_values = -jnp.abs((grid.states[...,0]%(2*jnp.pi)) - jnp.pi) + jnp.pi/2
# Solver Settings
times = np.linspace(0, -10, 101, endpoint=True)
solver_settings = hj.SolverSettings.with_accuracy("very_high", 
                                                  hamiltonian_postprocessor=
                                                  hj.solver.backwards_reachable_tube)

values = hj.solve(solver_settings,dynamics, grid, times, failure_values)
'''
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

"""
Use this helper function to save a GIF of your 
computed value function for Problem 3.2.
Please read the docstring below for what inputs you need to supply.
"""
def save_values_gif(values, grid, times, save_path='outputs/values.gif'):
    """
    args:
        values: ndarray with shape [
                len(times),
                len(grid.coordinate_vectors[0]),
                len(grid.coordinate_vectors[1])
            ]
        grid: hj.Grid
        times: ndarray with shape [len(times)]
    """
    vbar = 3
    fig, ax = plt.subplots()
    ax.set_title(f'$V(x, {times[0]:3.2f})$')
    ax.set_xlabel('$\\theta$ (rad)')
    ax.set_ylabel('$\\dot{\\theta}$ (rad/s)')
    value_plot = ax.pcolormesh(
        grid.coordinate_vectors[0],
        grid.coordinate_vectors[1],
        values[0].T,
        cmap='RdBu',
        vmin=-vbar,
        vmax=+vbar
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

#save_values_gif(values, grid, times, save_path='outputs/values.gif')


