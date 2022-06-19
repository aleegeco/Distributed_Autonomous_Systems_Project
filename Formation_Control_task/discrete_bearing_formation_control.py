import numpy as np
import networkx as nx
import control as ctrl
import matplotlib.pyplot as plt
from functions import *

# graph is time-varying, but let's build the simplest case to try the algorithm and then move formard
NN = 4
d = 2
n_leaders = 2

# Initialize Bearing Laplacian
B = np.zeros((d*NN,d*NN))
# it stores all the vector position for each node
# to extract a specific vector position you should do Node_pos[i-th node,:,:]
Node_pos = np.zeros((NN,d,1)) # it is a tensor with NN matrix of dimension (dxd)

# try to reproduce a rectangle
# Node 0, ( px py ).T
Node_pos[0,:,:] = np.array([[
    1,5]
    ]).T
# Node 1 (px py).T
Node_pos[1,:,:] = np.array([
    [5,5]
    ]).T
# Node 2 (px py).T
Node_pos[2,:,:] = np.array([[
    1,1]
    ]).T
# Node 3 (px py).T
Node_pos[3,:,:] = np.array([
    [5,1]
    ]).T

# it stores all the matrices Pg_ij ( d x d )
# to extract Pg_ij you must Pg_stack[node_i, node_j,:,:]
Pg_stack = np.zeros((NN,NN,d,d)) # it is a tensor of dimension 4 NN,NN of dimension (dxd)

# for cycles to pupulate the tensor Pg_stack
for node_i in range(NN):
    for node_j in range(NN):
        pos_j = Node_pos[node_j,:,:]
        pos_i = Node_pos[node_i,:,:]
        Pg_ij = proj_matrix(pos_i, pos_j)
        Pg_stack[node_i,node_j,:,:] = Pg_ij

## Pg_ij is always the same, but the Matrix B is computed according to the Adjacency Matrix of the graph
G = nx.binomial_graph(NN, 0.8)
Adj = nx.adjacency_matrix(G).toarray()
print(Adj)
fig = plt.figure()
nx.draw(G,with_labels= True)
plt.show()

B = bearing_dynamics(np.random.randn((d*NN)), Pg_stack, Adj, d)
print(B)
print(B.shape)


