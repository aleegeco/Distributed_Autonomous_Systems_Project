import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functions import *

# graph is time-varying, but let's build the simplest case to try the algorithm and then move formard
NN = 4  # number of agents
d = 2  # dimension of the pos/vel vector (2-dimensional or 3-dimension motion)
n_leaders = 2 # number of leaders
n_follower = NN - n_leaders # number of followers
dt = int(10e-4)
max_iter = 250 # number of iterations
horizon = np.linspace(0, max_iter, dtype=int)
p_plus = np.zeros((d*NN, 1, max_iter))
v_plus = np.zeros((d*NN, 1, max_iter))

p_plus[:,:,0] = np.random.rand((d*NN)).reshape([d*NN, 1])  # random initial conditions for position of all the agents
v_plus[:,:,0] = np.random.rand((d*NN)).reshape([d*NN, 1]) # random initial conditions for velocities of all the agents

B = np.zeros((d*NN, d*NN))  # Initialization of Bearing Laplacian

Node_pos = np.zeros((NN, d, 1))  # it stores all the vector position for each node
# it is a tensor with NN matrix of dimension (dxd)
# to extract a specific vector position you should do Node_pos[i-th node,:,:]

# try to reproduce a rectangle
# Node 0, ( px py ).T
Node_pos[0, :, :] = np.array([[
    1, 5]
    ]).T
# Node 1 (px py).T
Node_pos[1, :, :] = np.array([
    [5, 5]
    ]).T
# Node 2 (px py).T
Node_pos[2, :, :] = np.array([[
    1, 1]
    ]).T
# Node 3 (px py).T
Node_pos[3, :, :] = np.array([
    [5, 1]
    ]).T

# we impose the leaders positions
p_plus[[0, 1],:,0] = Node_pos[0, :, :]
p_plus[[2, 3],:,0] = Node_pos[1, :, :]

xx = np.zeros((2*d*NN, 1, max_iter))
xx[:,:,0] = np.concatenate((p_plus[:,:,0], v_plus[:,:,0]))

# Pg_ij is always the same, but the Matrix B is computed according to the Adjacency Matrix of the graph
G = nx.binomial_graph(NN, 0.8)
Adj = nx.adjacency_matrix(G).toarray()

Pg_stack = proj_stack(Node_pos,NN,d)
B = bearing_laplacian(Pg_stack, Adj, d)


fig = plt.figure()
nx.draw(G, with_labels=True)
plt.show()

Bff = B[d*n_leaders:d*NN, d*n_leaders:d*NN]
Bfl = B[d*n_leaders:d*NN, :d*n_leaders]

pos_leader = Node_pos[[0, 1], :, :].reshape([4, 1])
if np.linalg.det(Bff) != 0:
    print("The matrix Bff is not singular")
    pf_star = - np.linalg.inv(Bff)@Bfl@ pos_leader
else:
    print("The matrix Bff is singular! Check the Graph")
follow = [2, 3]
uu = np.ones((2, 1, max_iter))
for t in range(max_iter-1):
    xx[:,:,t+1] = bearing_dynamics(xx[:,:,t], uu[:,:,t], Adj, dt, d, follow)

figure = plt.figure()

# for time in range(max_iter):
#     plt.plot(xx[[i for i in range(NN) if i & 2 == 0],:,time],xx[range(0,NN,2),:,time])



