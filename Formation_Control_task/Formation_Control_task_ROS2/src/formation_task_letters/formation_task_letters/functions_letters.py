import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def bearing_vector(vec_pi: np.array, vec_pj: np.array, d=None):
    # vec_p is a position vector of dimension dx1
    # We have to check it
    row_i = np.shape(vec_pi)[0]
    row_j = np.shape(vec_pj)[0]
    if all(vec_pi == vec_pj): # if the vector is the same we have to avoid numerical error in the division
        g_ij = np.zeros((row_i, 1))
    else:
        if d:  # if we impose the dimension we'll enter this if, otherwise we'll move to the next statement
            if row_i == row_j and row_i == d:
                g_ij = (vec_pj - vec_pi)/np.linalg.norm(vec_pj - vec_pi)
            else:
                print("Vectors dimensions are con consistent")
        elif row_i == row_j and row_i == 2:  # we want to check if the two vectors have consistent dimension (fixed one)
            g_ij = (vec_pj - vec_pi)/np.linalg.norm(vec_pj - vec_pi)
        else:
            print("WARNING! The dimension of the vector are not consistent. Check them")
        # unit distance vector
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
    # it computes the matrix A_kron(G(p))
    n_agents = np.shape(Adj)[0]  # int representing number of agents
    B_temp = np.zeros((n_agents, n_agents, d, d))  # tensor of 4 dimension to build matrix A_kron
    for node_i in range(n_agents):
        list_neigh_i = np.nonzero(Adj[node_i])[0]  # neighbors of node i
        for node_j in range(n_agents):
            if node_j in list_neigh_i:  # if the node j is a neighbor
                if node_j != node_i:  # if the nodes are not the sa,e
                    B_temp[node_i, node_j, :, :] = - Pg_stack[node_i, node_j, :, :]
            elif node_i == node_j:  # if the node are the same (block diagonal of matrix A_kron)
                for node_k in list_neigh_i:
                    B_temp[node_i, node_j, :, :] += Pg_stack[node_i, node_k, :, :]  # summation of the matrices Pg_ik
    B = np.zeros((d * n_agents, d * n_agents))
    # another cycle to build the matrix A_kron with the right dimensions
    for i in range(n_agents):
        for j in range(n_agents):
            for k in range(d):
                for z in range(d):
                    B[i * d + k, j * d + z] = B_temp[i, j, k, z]
    return B


def kron_dynamical_matrix(B: np.array, NN: int, n_leaders: int, k_p: int, k_v: int, d: int):
    n_followers = NN - n_leaders
    zeros_nl = np.zeros((n_leaders, 2*NN))
    zeros_nf = np.zeros((n_followers, NN+n_leaders))
    I_nf = np.identity((n_followers))
    I_d = np.identity((d))

    Bff = B[d * n_leaders:d * NN, d * n_leaders:d * NN]
    Bfl = B[d * n_leaders:d * NN, :d * n_leaders]

    A = np.concatenate((zeros_nf, I_nf), axis=1)
    A = np.concatenate((zeros_nl, A), axis=0)
    A = np.concatenate((A, zeros_nl), axis=0)

    B_gains = np.concatenate((Bfl * k_p, Bff * k_p, Bfl * k_v, Bff * k_v), axis=1)

    A_kron = np.kron(A, I_d)
    A_kron = np.concatenate((A_kron, np.negative(B_gains)), axis=0)
    return A_kron


def formation(xx: np.array, horizon: int, Adj: np.array, NN: int, n_x: int, animate=True):
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

# function used in ROS2 to define the dynamics of each agent
def update_dynamics(dt: int, x_i: np.array, neigh: list, data, formation: np.array, agent_id: int,
                    n_leaders: int, k_p: int, k_v: int,integral_action=False):

    n_x = np.shape(x_i)[0]
    dd = n_x//2
    epsilon = 1e-2
    start = 0


    x_i = x_i.reshape([n_x,1])
    x_dot_i = np.zeros((n_x,1))

    pos_i = x_i[:dd]
    vel_i = x_i[dd:]
    vel_dot_i = np.zeros((dd, 1))

    error_pos = np.abs(pos_i - formation[0,agent_id,:].reshape((dd,1)))

    if agent_id < n_leaders:
        pos_dot_i = k_p*error_pos

        x_dot_i = np.concatenate((pos_dot_i, vel_dot_i))

        x_i = x_i + dt*x_dot_i

        if np.linalg.norm(error_pos) < epsilon:
            start = 1

        # for cycle to empty the buffer even if we're considering leaders, otherwise the algorithm will not converge
        for node_j in neigh:
            _ = np.array(data[node_j].pop(0)[1:-1]).reshape([n_x, 1])
    else:
        # for node_j in neigh:
        #     x_j = np.array(data[node_j].pop(0)[1:]).reshape([n_x,1])
        #     pos_j = x_j[:dd]
        #     vel_j = x_j[dd:]
        #
        #     if integral_action:
        #         pos_dot_i = vel_i
        #         vel_dot_i = vel_dot_i - k_p*Pg_stack_ii[node_j, :]@(pos_i - pos_j) \
        #                     - k_v*Pg_stack_ii[node_j, :]@(vel_i - vel_j)
        #     else:
        #         pos_dot_i = vel_i
        #         vel_dot_i = vel_dot_i - k_p * Pg_stack_ii[node_j, :] @ (pos_i - pos_j) \
        #                     - k_v * Pg_stack_ii[node_j, :] @ (vel_i - vel_j)
        #
        #     x_dot_i = np.concatenate((pos_dot_i, vel_dot_i))
        for node_j in neigh:
            _ = np.array(data[node_j].pop(0)[1:-1]).reshape([n_x, 1])
        x_dot_i = 0
        x_i = x_i + dt * x_dot_i

    return x_i, start
