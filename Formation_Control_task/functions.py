import numpy as np
import networkx as nx

def bearing_vector(vec_pi:np.array, vec_pj:np.array, d = None):
    # vec_p is a position vector of dimension dx1
    # We have to check it
    row_i = np.shape(vec_pi)[0]
    row_j= np.shape(vec_pj)[0]
    if d: # if we impose the dimension we'll enter this if, otherwise we'll move to the next statement
        if row_i == row_j and row_i == d:
            pass
        else:
            print("Vectors dimensions are con consistent")
    elif row_i == row_j and row_i == 2: # we want to check if the two vectors have consistent dimension (fixed one)
        pass
    else:
        print("WARNING! The dimension of the vector are not consistent. Check them")
    g_ij = (vec_pj - vec_pi) / np.linalg.norm(vec_pj - vec_pi) # unit distance vector
    return g_ij

def proj_matrix(vec_pi:np.array, vec_pj:np.array, d = None):
    # Function that computes the Projection Matrix based on the vectors p_i and p_j
    if d:
        g_ij = bearing_vector(vec_pi, vec_pj, d)
        Id = np.identity(d)
        Pg_ij = Id - g_ij@g_ij.T
    else:
        g_ij = bearing_vector(vec_pi, vec_pj)
        d1 = np.shape(g_ij)[0]
        Id = np.identity(d1)
        Pg_ij = Id - g_ij@g_ij.T
    return Pg_ij

def bearing_laplacian(Pg_stack:np.array, Adj:np.array, d:int):
    # function which computes the bearing laplacian
    # it computes the matrix B(G(p))
    n_agents = np.shape(Adj)[0] # int representing number of agents
    B_temp = np.zeros((n_agents,n_agents,d,d)) # tensor of 4 dimension to build matrix B
    for node_i in range(n_agents):
        list_neigh_i = np.nonzero(Adj[node_i])[0] # neighbors of node i
        for node_j in range(n_agents):
            if node_j in list_neigh_i: # if the node j is a neighbor
                if node_j != node_i: # if the nodes are not the sa,e
                    B_temp[node_i, node_j,:,:] = - Pg_stack[node_i, node_j,:,:]
            elif node_i == node_j: # if the node are the same (block diagonal of matrix B)
                for node_k in list_neigh_i:
                    B_temp[node_i, node_j,:,:] += Pg_stack[node_i, node_k,:,:] # summation of the matrices Pg_ik

    B = np.zeros((d*n_agents,d*n_agents))
    # another cycle to build the matrix B with the right dimensions
    for i in range(n_agents):
        for j in range(n_agents):
            for k in range(d):
                for z in range(d):
                    B[i*d+k,j*d+z] = B_temp[i,j,k,z]
    return B

def bearing_dynamics(p_t:np.array, v_t:np.array, B, dt):
    # function which computes the forward-euler discretization of the Bearing Laplacian Model
    n_p = np.shape(p_t)[0]
    n_v = np.shape(v_t)[0]
    if n_p == n_v: # check the dimension of position and velocity vector
        pos_plus = p_t + dt*(B@p_t)
        vel_plus = v_t + dt*(B@v_t)
        return pos_plus, vel_plus
    else:
        print("Error in Bearing Dynamics: Pos vector and Vel vector dimensions are not consistent")







