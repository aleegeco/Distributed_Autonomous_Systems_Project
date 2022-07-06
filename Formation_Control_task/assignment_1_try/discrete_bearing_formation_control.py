import numpy as np
import networkx as nx
import control as ctrl
import matplotlib.pyplot as plt
from functions import *

FIGURE = True
# graph is time-varying, but let's build the simplest case to try the algorithm and then move forward
NN = 4  # number of agents
dd = 2  # dimension of the pos/vel vector (2-dimensional or 3-dimension motion)
n_leaders = 2  # number of leaders
p_ER = 0.9
n_follower = NN - n_leaders  # number of followers

dt = 0.1  # discretion step
T_max = 50  # tempo massimo simulation
horizon = np.linspace(0.0, T_max, int(T_max / dt))
max_iter = 500  # number of iterations

# gain parameters

k_p = 0.7
k_v = 1.5

p_plus = np.zeros((dd * NN, 1, len(horizon)))
v_plus = np.zeros((dd * NN, 1, len(horizon)))

p_plus[:, :, 0] = np.random.rand((dd * NN)).reshape(
    [dd * NN, 1])  # random initial conditions for position of all the agents
v_plus[:, :, 0] = np.random.rand((dd * NN)).reshape(
    [dd * NN, 1])  # random initial conditions for velocities of all the agents

# B = np.zeros((d * NN, d * NN))  # Initialization of Bearing Laplacian

Node_pos = np.zeros((NN, dd, 1))  # it stores all the vector position for each node
# it is a tensor with NN matrix of dimension (dxd)
# to extract a specific vector position you should do Node_pos[i-th node,:,:]

# try to reproduce a rectangle
# Node 0, ( px py ).T
Node_pos[0, :, :] = np.array([
    [1, 5]
]).T
# Node 1 (px py).T
Node_pos[1, :, :] = np.array([
    [5, 5]
]).T
# Node 2 (px py).T
Node_pos[2, :, :] = np.array([
    [1, 1]
]).T
# Node 3 (px py).T
Node_pos[3, :, :] = np.array([
    [5, 1]
]).T

# we impose the leaders positions
p_plus[[0, 1], :, 0] = Node_pos[0, :, :]
p_plus[[2, 3], :, 0] = Node_pos[1, :, :]

# Pg_ij is always the same, but the Matrix B is computed according to the Adjacency Matrix of the graph
G = nx.binomial_graph(NN, p_ER)

Adj = nx.to_numpy_array(G)

Pg_stack = proj_stack(Node_pos, NN, dd)
B = bearing_laplacian(Pg_stack, Adj, dd)

if FIGURE:
    plt.figure(1)
    nx.draw(G, with_labels=True)
    plt.show()

Bff = B[dd * n_leaders:dd * NN, dd * n_leaders:dd * NN]
Bfl = B[dd * n_leaders:dd * NN, :dd * n_leaders]

pos_leader = Node_pos[[0, 1], :, :].reshape([4, 1])
if np.linalg.det(Bff) != 0:
    print("The matrix Bff is not singular")
    pf_star = - np.linalg.inv(Bff) @ Bfl @ pos_leader
else:
    print("The matrix Bff is singular! Check the Graph")

xx = np.zeros((NN * dd * 2, len(horizon)))
xx_star = np.zeros((NN * dd * 2, len(horizon)))

for i in range(NN):
    for j in range(dd):
        xx_star[(n_leaders * i) + j] = Node_pos[i, j, :]

# test to create the big matrix to define x_dot = A x
zeros_nl_2NN = np.zeros((n_leaders, 2 * NN))
zeros_nf_NN = np.zeros((n_follower, NN + n_leaders))
I_nf = np.identity(n_follower, dtype=int)
I_d = np.identity(dd, dtype=int)

A = np.concatenate((zeros_nf_NN, I_nf), axis=1)
A = np.concatenate((zeros_nl_2NN, A), axis=0)
A = np.concatenate((A, zeros_nl_2NN), axis=0)

# adding the Kp and Kv

temp = np.concatenate((Bfl * k_p, Bff * k_p, Bfl * k_v, Bff * k_v), axis=1)

A_kron = np.kron(A, I_d)

A_kron = np.concatenate((A_kron, np.negative(temp)), axis=0)
xx_init = np.zeros((NN * 2 * dd, 1))

for i in range(n_leaders):
    for j in range(dd):
        xx_init[(n_leaders * i) + j] = Node_pos[i, j, :]

xx[:, 0] = xx_init.reshape(16)

for tt in range(len(horizon) - 1):
    xx[:, tt + 1] = xx[:, tt] + dt * (A_kron @ xx[:, tt])
if FIGURE:
    plt.figure(2)
    for i in range(NN):
        plt.plot(horizon, xx[i * 2, :])
        plt.plot(horizon, xx_star[i * 2, :], "r--", linewidth=0.5)

    plt.figure(3)

    for i in range(NN):
        plt.plot(horizon, xx[i * 2 + 1, :])
        plt.plot(horizon, xx_star[i * 2 + 1, :], "r--", linewidth=0.5)

    plt.figure(4)

    for i in range(NN):
        plt.plot(xx[i * 2, :], xx[i * 2 + 1, :], "r--", linewidth=0.5)
        plt.plot(xx[i * 2, len(horizon) - 1], xx[i * 2 + 1, len(horizon) - 1], "ro")

print("DAJE TUTTO OK")
