import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as  hj
import cvxpy as cp
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from hj_reachability import dynamics
from hj_reachability import sets
from scipy.interpolate import interpn
from problem1_helper import save_values_gif, plot_value_and_safe_set_boundary

u_bar = 3.
d_lower = 0.
d_upper = 0.4

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
            [1.],
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

# Find the volume of the safe set
values_converged = values[-1]

# Save the values as a gif
#save_values_gif(values, grid, times)

# Finding the optimal controller
grads = grid.grad_values(values_converged, solver_settings.upwind_scheme)
beta2s = grads[:, :, 1]
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

def nominal_control(t, theta, theta_dot):
    l, m, g = 1, 2, 10
    if t < 1:
        return 3
    elif t < 2:
        return -3
    elif t < 3:
        return 3
    else:
        return m*l**2*(-g/l*np.sin(theta) - np.dot([1.5, 1.5], [theta, theta_dot]))
    
def qp_controller(t, theta, theta_dot, gamma, disturbance=0):
    # Define the variable
    u = cp.Variable(1)
    # Define the objective and constraints
    obj = cp.Minimize(cp.sum_squares(u - nominal_control(t, theta, theta_dot)))
    constr = [u >= -u_bar, u <= u_bar]

    f = dynamics.open_loop_dynamics(np.array([theta, theta_dot]), t)
    g = dynamics.control_jacobian(np.array([theta, theta_dot]), t)
    D = dynamics.disturbance_jacobian(np.array([theta, theta_dot]), t)

    dyn = f + g @ u + D * disturbance

    grad_v = grad_at_state(np.array([theta, theta_dot]))
    v = value_at_state(np.array([theta, theta_dot]))
    
    constr += [cp.matmul(grad_v, dyn) + gamma * v >= 0]
    prob = cp.Problem(obj, constr)
    prob.solve()
    return u.value.item() if u.value is not None else dynamics.optimal_control(np.array([theta, theta_dot]), 0.0, grad_v).item()

def value_at_state(x):
    v = interpn(
        ([np.array(v) for v in grid.coordinate_vectors]),  # grid vectors
        np.array(values_converged),                       # level-set array
        np.atleast_2d(x),                                 # shape (1, 2)
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    return v.item()

def grad_at_state(x):
    return np.array([
        interpn(
            points=[np.array(v) for v in grid.coordinate_vectors],
            values=np.array(grads[:,:,i]),
            xi=np.array(x),
            method='linear',
            bounds_error=False,
            fill_value=0.0
        ).item()
        for i in range(2)
    ])



T = 5
dt = 0.01
def simulate(x0, filter, gamma=0):
    nt = int(T / dt)
    xs = np.full((nt, 2), fill_value=np.nan)
    us = np.full((nt, 1), fill_value=np.nan)
    xs[0] = x0
    for i in range(1, nt):
        x = xs[i-1]
        disturbance = np.random.uniform(d_lower, d_upper)
        value_x = value_at_state(x)
        if value_x > 0 and filter !=3:
            u = nominal_control(i*dt, x[0], x[1])
        else:
            if filter == 1:
                # Least restrictive filter
                u = optimal_safety_controller(x)
                u = u_bar * u
            elif filter == 2:
                # Smooth least restrictive filter
                u = qp_controller(i*dt, x[0], x[1], 0, disturbance)
            elif filter == 3:
                # Smooth blending filter
                u = qp_controller(i*dt, x[0], x[1], gamma, disturbance)
        us[i-1] = u
        xs[i] = x + dt*np.array([x[1] + disturbance, 10/1*jnp.sin(x[0]) + u*1/(2)])
    return xs, us

# Simulate the system 100 times
safe_count = 0

for i in range(100):
    # Simulate the system with the disturbance
    x0 = np.array([0.05, 0.05])
    xs, us = simulate(x0, filter=3, gamma = 0.5)

    # Check if safe (all values[xs] > 0)
    safe = []
    for _,x in enumerate(xs):
        value_x = value_at_state(x)
        if value_x < -0.5:
            safe.append(False)
        else:
            safe.append(True)

    if all(safe):
        safe_count += 1

plt1, ax1 = plt.subplots()
ax1.plot(xs[:, 0], xs[:, 1], linewidth=2, color='purple')
plt.title(f'Optimally Safe Trajectory from $theta={0.05}$, $theta dot={0.05}$')
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
plt1.savefig('outputs/P2a_results.png',            # or .pdf, .svg, .jpg, etc.
        dpi=300,                   # resolution in dots per inch
        bbox_inches="tight")       # crop to the content


plt1, ax1 = plt.subplots()
plt.title(f'Optimally Safe Controller from $theta={0.05}$, $theta dot={0.05}$')
plt.xlabel('$Time, t$ (s)')
plt.ylabel('$Control Input, u$ (rad/s)')
time = np.linspace(0, T, int(T/dt), endpoint=True)
ax1.plot(time, us, linewidth=2, color='purple')
plt1.savefig('outputs/P2a_results_control.png',            # or .pdf, .svg, .jpg, etc.
        dpi=300,                   # resolution in dots per inch
        bbox_inches="tight")       # crop to the content

# Print the safety rate
print(f"Safety rate: {safe_count:.2f}%")
