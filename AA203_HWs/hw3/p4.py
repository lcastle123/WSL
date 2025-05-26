import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

A = np.array([[0.9, 0.6],[0,0.8]])
B = np.array([[0],[1]])
P = np.eye(2)
N = 4
r_x = 5
r_u = 1
Q = np.eye(2)
R = np.eye(1)
T = 15

# Define X_T by solving the convex problem

M = cp.Variable((2, 2), PSD=True)

objective = cp.Maximize(cp.log_det(M))
constraints = [
    M >> 0,
    M <= r_x**2 * np.eye(2),
    cp.bmat([[M, cp.matmul(A,M)], [cp.matmul(M,A.T), M]]) >> 0,
]

problem = cp.Problem(objective, constraints)
problem.solve()

def generate_ellipsoid_points(M, num_points: int = 100) -> np.ndarray:
    """
    Generate points on the boundary of a 2-D ellipsoid.

    The ellipsoid is defined by
        { x ∈ ℝ² | x.T @ inv(M) @ x ≤ 1 },
    where `M` is a symmetric positive-definite 2x2 matrix.
    """
    L = np.linalg.cholesky(M)
    theta = np.linspace(0.0, 2.0 * np.pi, num_points)
    u = np.column_stack((np.cos(theta), np.sin(theta)))
    x = u @ L.T
    return x


# Generate points on the boundary of the ellipsoid
x_T = generate_ellipsoid_points(M.value, num_points=100)
# Plot the ellipsoid
plt.figure()
plt.plot(x_T[:, 0], x_T[:, 1], label="X_T Ellipsoid Boundary")
plt.xlim(-r_x, r_x)
plt.ylim(-r_x, r_x)
plt.xlabel("x1")
plt.ylabel("x2")

# Find the full state space, which is a cicle with radius r_x
theta = np.linspace(0.0, 2.0 * np.pi, 100)
x = r_x * np.cos(theta)
y = r_x * np.sin(theta)
plt.plot(x, y, label="Full State Space")

# Plot A time x_T
Ax_T = A @ x_T.T
plt.plot(Ax_T[0, :], Ax_T[1, :], label="A * X_T Ellipsoid Boundary")

plt.legend()
plt.title("Ellipsoid Boundaris in Full State Space")
plt.grid()
plt.axis("equal")
plt.savefig("ellipsoid_boundary.png")


# Print the value of M^-1 to 3 decimal places
print("W:=M^-1:")
print(np.round(np.linalg.inv(M.value), 3))
W = np.linalg.inv(M.value)


n, m = Q.shape[0], 1
x_cvx = cp.Variable((N + 1, n))
u_cvx = cp.Variable((N, m))
x0 = cp.Parameter((n,), value=np.zeros(n))

cost = 0.0
constraints = []

for k in range(N):
    # Cost function
    cost += cp.quad_form(x_cvx[k], Q) + cp.quad_form(u_cvx[k], R)
    # Dynamics constraints
    constraints.append(x_cvx[k + 1] == A @ x_cvx[k] + B @ u_cvx[k])
    # State constraints
    constraints.append(cp.norm(x_cvx[k], "inf") <= r_x)
    # Input constraints
    constraints.append(cp.norm(u_cvx[k], "inf") <= r_u)
# Intial state constraint
constraints.append(x_cvx[0] == x0)
# Terminal state constraint
constraints.append(cp.quad_form(x_cvx[N], W) <= 1)  
# Terminal state cost
cost += cp.quad_form(x_cvx[N], P)

init_x0 = np.array([0, -4.5])

fig, ax = plt.subplots(2, dpi=150, figsize=(10, 8), sharex="row", sharey="row")
x_mpc = np.zeros((T, N + 1, 2))
u_mpc = np.zeros((T, N, 1))
x0.value = init_x0
for t in range(T):
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(cp.CLARABEL)
    x_mpc[t] = x_cvx.value
    u_mpc[t] = u_cvx.value
    status = prob.status
    if status == "infeasible":
        x_mpc = x_mpc[:t]
        u_mpc = u_mpc[:t]
        break
    x0.value = A @ x0.value + B @ u_mpc[t, 0, :]
    ax[0].plot(x_mpc[t, :, 0], x_mpc[t, :, 1], "--*", color="k")
    if t == 0:
        ax[0].plot(x_mpc[t, 0, 0], x_mpc[t, 0, 1], "--*", color="k", label="Planned Trajectory")
ax[0].plot(x_mpc[:, 0, 0], x_mpc[:, 0, 1], "-o", label="Actual Trajectory")
ax[1].plot(u_mpc[:, 0], "-o")
ax[0].set_title("State Trajectory")
plt.subplots_adjust(hspace=0.4)
ax[1].set_title("Control Input")
ax[0].set_xlabel(r"$x_{k,1}$")
ax[1].set_xlabel(r"$k$")
ax[0].set_ylabel(r"$x_{k,2}$")
ax[1].set_ylabel(r"$u_k$")

# Plot the ellipsoid
ax[0].plot(x_T[:, 0], x_T[:, 1], label="X_T Ellipsoid Boundary")
# Find the full state space, which is a cicle with radius r_x
theta = np.linspace(0.0, 2.0 * np.pi, 100)
x = r_x * np.cos(theta)
y = r_x * np.sin(theta)
ax[0].plot(x, y, label="Full State Space")
# Plot A time x_T
Ax_T = A @ x_T.T
ax[0].plot(Ax_T[0, :], Ax_T[1, :], label="A * X_T Ellipsoid Boundary")
ax[0].legend()
ax[0].axis("equal")
ax[0].grid()
fig.savefig("mpc_trajectories.png", bbox_inches="tight")
