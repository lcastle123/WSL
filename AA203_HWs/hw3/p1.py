import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

n = 20 # Size of grid
sigma = 10 # Probability constant
gamma = 0.95 # Discount factor
x_eye = np.array([15,15])
x_goal = np.array([19,9])
action_dict = np.array([ [0, 1], [0, -1], [-1, 0], [1, 0]])

# Helper functions to convert between coordinates and indices
idx = lambda x: x[0] * n + x[1]
coord = lambda k: np.array([k // n, k % n])

omega = np.zeros((n*n))
for i in range(n*n):
    omega[i] = np.exp(-np.linalg.norm(coord(i) - x_eye)**2 / (2 * sigma**2))

def state_change(s, a):
    # Give the next state given the current state and action and make sure it stays in the grid
    sp = s + action_dict[a]
    if sp[0] < 0 or sp[0] > n-1 or sp[1] < 0 or sp[1] > n-1:
        return s
    else:
        return sp

def get_reward(s, a):
    if np.all(s == x_goal):
        return 1
    else:
        return 0
    

# Iterate the value function
V = np.zeros((n*n))
V_new = np.zeros((n*n))
V_new = np.zeros((n*n))
epsilon = np.inf
iter = 0
while epsilon > 1e-5 and iter < 2000:
    for i in range(n*n):
        a_list = np.zeros((4))
        for a in range(4):
            for sp_opt in range(4):
                p = omega[i] / 4
                if sp_opt == a:
                    p += (1 - omega[i])
                sp = state_change(coord(i), sp_opt)
                a_list[a] += p *(get_reward(coord(i), sp_opt) + gamma * V[idx(sp)])
        V_new[i] = np.max(a_list)
    epsilon = np.max(np.abs(V_new - V))
    V = V_new.copy()
    iter += 1
print(iter, " , ",epsilon)

# Plot the value function
V_opt = V.reshape(n, n) 
plt.imshow(V_opt.T, origin='lower')
plt.colorbar(label='V*')
plt.scatter(*x_goal, marker='*', s=120, c='green', label='Goal')
plt.scatter(*x_eye, marker='o', s=120, c='red', label='Eye')
plt.title('Optimal value function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('value_function.png')


# b) 

# Find the optimal policy
policy = np.zeros((n*n))
for i in range(n*n):
    a_list = np.zeros((4))
    for a in range(4):
        for sp_opt in range(4):
            p = omega[i] / 4
            if sp_opt == a:
                p += (1 - omega[i])
            sp = state_change(coord(i), sp_opt)
            a_list[a] += p *(get_reward(coord(i), sp_opt) + gamma * V[idx(sp)])
    policy[i] = np.argmax(a_list)

# Simulate the policy
x_init = np.array([0,19])
trajectory = np.zeros((100,2))
x = x_init.copy()
trajectory[0] = x
for i in range(100-1):
    a = policy[idx(x)]
    omega_val = omega[idx(x)]
    if np.random.rand() < omega_val:
        a = np.random.randint(0, 4)
    x = state_change(x, int(a))
    trajectory[i+1] = x
    if np.all(x == x_goal):
        break
trajectory = trajectory[:i+2]

# Plot the policy and trajectory
hex_colors = ["#DB13F5", "#750195","#f07a20","#70b3a8"]
cmap = ListedColormap(hex_colors)
policy_opt = policy.reshape(n, n) 
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(policy_opt.T, origin='lower', cmap=cmap, vmin=0, vmax=3)
labels = ['Up', 'Down', 'Left', 'Right']
handles = [Patch(facecolor=cmap(i), edgecolor='k', label=lbl) for i, lbl in enumerate(labels)]
ax.scatter(*x_goal, marker='*', s=120, c='green', label='Goal')
ax.scatter(*x_eye,  marker='o', s=120, c='red',   label='Storm eye')
ax.plot(trajectory[:,0], trajectory[:,1], '-', color='black', lw=1.5, label='Trajectory')
h_exist, l_exist = ax.get_legend_handles_labels()     # from goal, eye, trajectory
labels  = ['Up', 'Down', 'Left', 'Right']
handles = [Patch(facecolor=cmap(i), edgecolor='k', label=lbl)
           for i, lbl in enumerate(labels)]
ax.legend(handles + h_exist, labels + l_exist, loc='upper left', title='Legend')
ax.set_title('Optimal policy and simulated trajectory')
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
plt.tight_layout()
plt.savefig('policy_trajectory.png')
