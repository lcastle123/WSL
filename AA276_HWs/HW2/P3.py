import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as  hj

from hj_reachability import dynamics
from hj_reachability import sets

u_bar = 3.

class InvertedPendulum(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, m=2., l=1., g=10, u_bar=3.,
                 control_mode="max", disturbance_mode="min",
                 control_space=None, disturbance_space=None):
        self.m = m
        self.l = l
        self.g = g
        self.u_bar = u_bar
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-u_bar]), jnp.array([u_bar]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([0.]), jnp.array([0.]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        theta, theta_dot = state
        return jnp.array([theta_dot, self.g / self.l * jnp.sin(theta)])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [1 / (self.l*self.l*self.m)]
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.]
        ])
    
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(np.array([-np.pi, -10]), 
                np.array([np.pi, 10])),
    (101, 101))

# Define the failure set

failure_values = 0.3 - jnp.abs(grid.states[...,0])

# Solver Settings
times = np.linspace(0, -5, 101, endpoint=True)
solver_settings = hj.SolverSettings.with_accuracy("very_high", 
                                                  hamiltonian_postprocessor=
                                                  hj.solver.backwards_reachable_tube)

dynamics = InvertedPendulum(u_bar=u_bar)
values = hj.solve(solver_settings,dynamics, grid, times, failure_values)

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


"""
Use this helper function to visualize the value function 
and safe set boundary appropriately for Problem 3.4.
Please read the docstring below for what inputs you need to supply.
"""
def plot_value_and_safe_set_boundary(values_converged, grid, ax):
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
        ([np.array(v) for v in grid.coordinate_vectors]),
        np.array(values_converged),
        bounds_error=False,
        fill_value=None
    )
    vbar=3
    vis_thetas = np.linspace(-0.5, +0.5, num=101, endpoint=True)
    vis_theta_dots = np.linspace(-1, +1, num=101, endpoint=True)
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

# save_values_gif(values, grid, times, save_path='outputs/values.gif')

# Find the volume of the safe set
values_converged = values[-1]
safe_safe = values_converged > 0

# Number of 1s in safe_set
safe_set_size = np.sum(safe_safe)
# Total size of the grid
total_size = 101*101

# Total volume of the grid
total_volume = (np.pi-(-np.pi)) * (10-(-10))
# Volume of the safe set
safe_set_volume = safe_set_size / total_size * total_volume
print(f"Volume of the safe set: {safe_set_volume:.9f}")


# Finding the optimal controller
grads = grid.grad_values(values_converged, solver_settings.upwind_scheme)
beta2s = grads[:, :, 1]

from scipy.interpolate import interpn
def optimal_safety_controller(x):
  beta2 = interpn(
      ([np.array(v) for v in grid.coordinate_vectors]),
      np.array(beta2s),
      x,
      method='linear',
      bounds_error=False,
      fill_value=None
  )
  return np.sign(beta2).item()


T = 1
dt = 0.01
def simulate(x0):
  nt = int(T / dt)
  xs = np.full((nt, 2), fill_value=np.nan)
  us = np.full((nt, 1), fill_value=np.nan)
  xs[0] = x0
  for i in range(1, nt):
    x = xs[i-1]
    u = optimal_safety_controller(x)
    u = u_bar * u
    us[i] = u
    xs[i] = x + dt*np.array([x[1], 10/1*jnp.sin(x[0]) + u*1/(2)])
  return xs, us

def plot_trajectory(theta, theta_dot, 
                    save_path='outputs/P3d_results.png', 
                    values_converged=values_converged, grid=grid):
  xs,_ = simulate(np.array([theta, theta_dot]))
  plt1, ax1 = plt.subplots()
  ax1.plot(xs[:, 0], xs[:, 1], linewidth=2, color='purple')
  plt.title(f'Optimally Safe Trajectory from $theta={theta}$, $theta dot={theta_dot}$')
  plt.xlabel('$theta$ (rad)')
  plt.ylabel('$theta dot$ (rad/s)')
  plot_value_and_safe_set_boundary(values_converged, grid, ax1)
  ax1.contour(
      grid.coordinate_vectors[0],
      grid.coordinate_vectors[1],
      failure_values.T,
      levels=0,
      colors='r'
  )
  plt.xlim(-0.5, 0.5)
  plt.ylim(-1., 1.)
  plt1.savefig(save_path,            # or .pdf, .svg, .jpg, etc.
            dpi=300,                   # resolution in dots per inch
            bbox_inches="tight")       # crop to the content


# Example usage
plot_trajectory(-0.1, 0.4, 'outputs/P3d_results(-0.1,0.4).png')
plot_trajectory(-0.1, -0.3, 'outputs/P3d_results(-0.1,-0.3).png')

def plot_control(theta, theta_dot, save_path='outputs/P3d_results_control.png'):
  _, us = simulate(np.array([theta, theta_dot]))
  plt1, ax1 = plt.subplots()
  plt.title(f'Optimally Safe Controller from $theta={theta}$, $theta dot={theta_dot}$')
  plt.xlabel('$Time, t$ (s)')
  plt.ylabel('$Control Input, u$ (rad/s)')
  time = np.linspace(0, 1, 100, endpoint=True)
  ax1.plot(time, us, linewidth=2, color='purple')
  plt1.savefig(save_path,            # or .pdf, .svg, .jpg, etc.
            dpi=300,                   # resolution in dots per inch
            bbox_inches="tight")       # crop to the content

plot_control(-0.1, 0.4, 'outputs/P3d_results_control(-0.1,0.4).png')
plot_control(-0.1, -0.3, 'outputs/P3d_results_control(-0.1,-0.3).png')
