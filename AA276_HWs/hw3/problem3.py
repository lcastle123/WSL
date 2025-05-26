import torch
import cvxpy as cp
from problem3_helper import control_limits, f, g

from problem3_helper import NeuralVF
vf = NeuralVF()

# environment setup
obstacles = torch.tensor([
    [1.0,  0.0, 0.5], # [px, py, radius]
    [4.0,  2.0, 1.0],
    [4.0, -2.0, 1.0],
    [7.0,  0.0, 1.5],
    [7.0,  4.0, 0.5],
    [7.0, -4.0, 0.5]
])

def min_val_val(x):
    # Finds the value function with each individual obstacle, and then returns 
    # the lowest one, with the associated obstacle and gradient
    
    x_modified = torch.zeros((obstacles.shape[0], x.shape[0]), device=x.device)
    for i in range(obstacles.shape[0]):
        x_modified[i, :] = x
        x_modified[i, 0] = (x_modified[i, 0] - obstacles[i, 0])*0.5/obstacles[i, 2]
        x_modified[i, 1] = (x_modified[i, 1] - obstacles[i, 1])*0.5/obstacles[i, 2]
    vals = vf.values(x_modified)
    grads = vf.gradients(x_modified)
    min_val = torch.min(vals)
    min_index = torch.argmin(vals)
    grad = grads[min_index, :]
    return min_val, grad

def smooth_blending_safety_filter(x, u_nom, gamma, lmbda):
    """
    Compute the smooth blending safety filter.
    Refer to the definition provided in the handout.
    You might find it useful to use functions from
    previous homeworks, which we have imported for you.
    These include:
      control_limits(.)
      f(.)
      g(.)
      vf.values(.)
      vf.gradients(.)
    NOTE: some of these functions expect batched inputs,
    but x, u_nom are not batched inputs in this case.
    
    args:
        x:      torch tensor with shape [13]
        u_nom:  torch tensor with shape [4]
        
    returns:
        u_sb:   torch tensor with shape [4]
    """
    # YOUR CODE HERE

    u_sb = cp.Variable(4)
    s = cp.Variable(1)
    u_bar = control_limits()
    u_upper, u_lower = u_bar

    obj = cp.Minimize(cp.sum_squares(u_sb - u_nom) + lmbda * cp.square(s))
    constr = [u_sb >= u_lower, u_sb <= u_upper]

    f_val = f(x.unsqueeze(0)).squeeze(0)
    g_val = g(x.unsqueeze(0)).squeeze(0)
    
    dyn = f_val + g_val @ u_sb

    v, grad_v = min_val_val(x)
    
    constr += [cp.matmul(grad_v, dyn) + gamma * v + s >= 0]
    constr += [s >= 0]
    prob = cp.Problem(obj, constr)
    prob.solve()
    return torch.tensor(u_sb.value, dtype=torch.float32) # NOTE: ensure you return a float32 tensor
