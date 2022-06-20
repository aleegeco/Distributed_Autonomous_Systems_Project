import numpy as np
import networkx as nx
import control as ctrl
import matplotlib.pyplot as plt
from functions import *

# graph is time-varying, but let's build the simplest case to try the algorithm and then move formard
NN = 4  # number of agents
d = 2  # dimension of the pos/vel vector (2-dimensional or 3-dimension motion)
n_leaders = 2
dt = int(10e-4)


P0 = np.random.rand((d*NN))  # random initial conditions for position and velocity of all the agents
V0 = np.random.rand((d*NN))
# NB X = [ p, v].T where p is a vector d*N and v is a vector d*N, x is a vector 2*d*N

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

Pg_stack = np.zeros((NN, NN, d, d))  # it stores all the matrices Pg_ij ( d x d )
# it is a tensor of dimension 4 NN,NN of dimension (dxd)
# to extract Pg_ij you must Pg_stack[node_i, node_j,:,:]

# for cycles to populate the tensor Pg_stack
for node_i in range(NN):
    for node_j in range(NN):
        pos_j = Node_pos[node_j, :, :]
        pos_i = Node_pos[node_i, :, :]
        Pg_ij = proj_matrix(pos_i, pos_j)
        Pg_stack[node_i, node_j, :, :] = Pg_ij

# Pg_ij is always the same, but the Matrix B is computed according to the Adjacency Matrix of the graph
G = nx.binomial_graph(NN, 0.8)
Adj = nx.adjacency_matrix(G).toarray()
print(Adj)
fig = plt.figure()
nx.draw(G, with_labels=True)
plt.show()

B = bearing_laplacian(Pg_stack, Adj, d)
p_plus, v_plus = bearing_dynamics(P0, V0, B, dt)

print(B)
print(B.shape)
print("p_t+1 ={}".format(p_plus), "v_t+1={}".format(v_plus))
