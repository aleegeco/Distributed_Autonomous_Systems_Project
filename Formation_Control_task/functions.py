import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def bearing_vector(vec_pi: np.array, vec_pj: np.array, d=None):
    # vec_p is a position vector of dimension dx1
    # We have to check it
    row_i = np.shape(vec_pi)[0]
    row_j = np.shape(vec_pj)[0]
    if d:  # if we impose the dimension we'll enter this if, otherwise we'll move to the next statement
        if row_i == row_j and row_i == d:
            pass
        else:
            print("Vectors dimensions are con consistent")
    elif row_i == row_j and row_i == 2:  # we want to check if the two vectors have consistent dimension (fixed one)
        pass
    else:
        print("WARNING! The dimension of the vector are not consistent. Check them")
    g_ij = np.divide((vec_pj - vec_pi), np.linalg.norm(vec_pj - vec_pi))  # unit distance vector
    return g_ij


def proj_matrix(vec_pi: np.array, vec_pj: np.array, d=None):
    # Function that computes the Projection Matrix based on the vectors p_i and p_j
    if d:
        g_ij = bearing_vector(vec_pi, vec_pj, d)
        Id = np.identity(d)
        Pg_ij = Id - g_ij @ g_ij.T
    else:
        g_ij = bearing_vector(vec_pi, vec_pj)
        d1 = np.shape(g_ij)[0]
        Id = np.identity(d1)
        Pg_ij = Id - g_ij @ g_ij.T
    return Pg_ij


def proj_stack(pos_nodes: np.array, NN: int, d: int):
    # function to define a tensor who store all projection matrices for the nodes
    Pg_stack = np.zeros((NN, NN, d, d))  # tensor of dimension (NN, NN, d,d)
    for node_i in range(NN):
        for node_j in range(NN):
            pos_j = pos_nodes[node_j, :, :]
            pos_i = pos_nodes[node_i, :, :]
            Pg_ij = proj_matrix(pos_i, pos_j)
            Pg_stack[node_i, node_j, :, :] = Pg_ij
    return Pg_stack


def bearing_laplacian(Pg_stack: np.array, Adj: np.array, d: int):
    # function which computes the bearing laplacian
    # it computes the matrix B(G(p))
    n_agents = np.shape(Adj)[0]  # int representing number of agents
    B_temp = np.zeros((n_agents, n_agents, d, d))  # tensor of 4 dimension to build matrix B
    for node_i in range(n_agents):
        list_neigh_i = np.nonzero(Adj[node_i])[0]  # neighbors of node i
        for node_j in range(n_agents):
            if node_j in list_neigh_i:  # if the node j is a neighbor
                if node_j != node_i:  # if the nodes are not the sa,e
                    B_temp[node_i, node_j, :, :] = - Pg_stack[node_i, node_j, :, :]
            elif node_i == node_j:  # if the node are the same (block diagonal of matrix B)
                for node_k in list_neigh_i:
                    B_temp[node_i, node_j, :, :] += Pg_stack[node_i, node_k, :, :]  # summation of the matrices Pg_ik
    B = np.zeros((d * n_agents, d * n_agents))
    # another cycle to build the matrix B with the right dimensions
    for i in range(n_agents):
        for j in range(n_agents):
            for k in range(d):
                for z in range(d):
                    B[i * d + k, j * d + z] = B_temp[i, j, k, z]
    return B


def bearing_dynamics(p_t: np.array, v_t: np.array, B: np.array, dt: int):
    # function which computes the forward-euler discretization of the Bearing Laplacian Model
    n_p = np.shape(p_t)[0]
    n_v = np.shape(v_t)[0]
    if n_p == n_v:  # check the dimension of position and velocity vector
        pos_plus = p_t + dt * (B @ p_t)
        vel_plus = v_t + dt * (B @ v_t)
        return pos_plus, vel_plus
    else:
        print("Error in Bearing Dynamics: Pos vector and Vel vector dimensions are not consistent")


def formation(xx, horizon, Adj, NN, n_x, animate=True):
    TT = np.size(horizon, 0)
    for tt in range(np.size(horizon, 0)):
        xx_tt = xx[:, tt].T
        for ii in range(NN):
            for jj in range(NN):
                index_ii = ii * n_x + np.array(range(n_x))
                p_prev = xx_tt[index_ii]
                plt.plot(p_prev[0], p_prev[1], marker='o', markersize=20, fillstyle='none')
                if Adj[ii, jj] & (jj > ii):
                    index_jj = (jj % NN) * n_x + np.array(range(n_x))
                    p_curr = xx_tt[index_jj]
                    plt.plot([p_prev[0], p_curr[0]], [p_prev[1], p_curr[1]],
                             linewidth=2, color='tab:blue', linestyle='solid')

        axes_lim = (np.min(xx) - 1, np.max(xx) + 1)
        plt.xlim(axes_lim)
        plt.ylim(axes_lim)
        plt.plot(xx[0:n_x * NN:n_x, :].T, xx[1:n_x * NN:n_x, :].T)
        plt.axis('equal')

        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()
