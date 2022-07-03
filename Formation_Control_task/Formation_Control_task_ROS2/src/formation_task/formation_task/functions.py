import numpy as np
import networkx as nx
from roboticstoolbox.tools.trajectory import *


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
    if d:  # if we impose a different dimension
        g_ij = bearing_vector(vec_pi, vec_pj, d)  # recall the function of bearing vector
        Id = np.identity(d)
        Pg_ij = Id - g_ij @ g_ij.T  # compute the Pg_ij
    else:  # if we do not impose dimension
        g_ij = bearing_vector(vec_pi, vec_pj)
        d1 = np.shape(g_ij)[0]
        Id = np.identity(d1)
        Pg_ij = Id - g_ij @ g_ij.T  # compute the Pg_ij
    return Pg_ij


def proj_stack(pos_nodes: np.array, NN: int, d: int):
    # function to define a tensor who store all projection matrices for each pair nodes
    Pg_stack = np.zeros((NN, NN, d, d))  # tensor of dimension (NN, NN, d,d)
    for node_i in range(NN):
        for node_j in range(NN):
            pos_j = pos_nodes[node_j, :, :]
            pos_i = pos_nodes[node_i, :, :]
            Pg_ij = proj_matrix(pos_i, pos_j)
            Pg_stack[node_i, node_j, :, :] = Pg_ij  # it stores for each pair of nodes (i,j) the associated proj. matrix
    return Pg_stack  # return the store of all the Pg_ij for each pair of nodes


def bearing_laplacian(Pg_stack: np.array, Adj: np.array, d: int):
    # function which computes the bearing laplacian
    # it computes the matrix B(G(p*))
    n_agents = np.shape(Adj)[0]  # int representing number of agents
    B_temp = np.zeros((n_agents, n_agents, d, d))  # temp tensor of 4  dimension to create the matrix B(G(p*))
    for node_i in range(n_agents):
        list_neigh_i = np.nonzero(Adj[node_i])[0]  # neighbors of node i
        for node_j in range(n_agents):
            if node_j in list_neigh_i:  # if the node j is a neighbor
                if node_j != node_i:  # if the nodes are not the same
                    B_temp[node_i, node_j, :, :] = - Pg_stack[node_i, node_j, :, :]
            elif node_i == node_j:  # if the node are the same
                for node_k in list_neigh_i:
                    B_temp[node_i, node_j, :, :] += Pg_stack[node_i, node_k, :, :]  # summation of the matrices Pg_ik
    # now we explode the matrix B_temp by imposing the right dimension for B(G(p*))
    B = np.zeros((d * n_agents, d * n_agents))
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


def update_dynamics(dt: int, self):
    # function which update the dynamics of each agent in ROS2, it is computed agent-wise not in compact matrix form
    # it takes "self" so we can use every class variable without explicitly declare it

    n_x = np.shape(self.x_i)[0] # dimension of the state vector
    dd = n_x//2 # dimension of position and velocity vector (i.e. we're in the plane XY or in the space XYZ)
    K_i = np.zeros((dd,dd))
    x_i = self.x_i # state of the agent i at this time step
    x_dot_i = np.zeros(n_x)
    # divide the dynamics in position and velocity vector
    pos_i = x_i[:dd]
    vel_i = x_i[dd:]
    vel_dot_i = np.zeros(dd)
    w_i = np.ones(dd) * 0.1 # constant input disturbance

    if self.agent_id < self.n_leaders: # if the considered agent is a leader
        if self.leader_acceleration: # if we impose an acceleration to leaders
            pos_dot_i = vel_i
            vel_dot_i = piecewise_acc(self)
            x_dot_i = np.concatenate((pos_dot_i, vel_dot_i))
            x_i = x_i + dt*x_dot_i

            for node_j in self.neigh:  # for cycle to empty the buffer even if we're considering leaders, otherwise the algorithm will not continue
                x_j = np.array(self.received_data[node_j].pop(0)[1:])
                # vel_j = x_j[dd:]
                # index_i = node_j * dd + np.arange(dd)
                # self.store_acc[index_i, self.tt] = vel_j  # store neighbors velocities to compute their derivatives
        else: # otherwise, the leader remains still
            x_i = x_i
            for node_j in self.neigh:  # for cycle to empty the buffer even if we're considering leaders,
                                        # otherwise the algorithm will not continue
                _ = np.array(self.received_data[node_j].pop(0)[1:])

    else: # if we are not leaders we'll enter this else
        for node_j in self.neigh:
            K_i += self.Pg_stack_ii[node_j, :]

        for node_j in self.neigh:
            x_j = np.array(self.received_data[node_j].pop(0)[1:]) # I take the received state from the message
            pos_j = x_j[:dd]
            vel_j = x_j[dd:]

            index_i = node_j*dd + np.arange(dd)
            self.store_acc[index_i, self.tt] = vel_j  # store the acceleration to compute the derivative
            self.error_pos[self.agent_id, node_j, :] += (pos_i - pos_j)*dt  # increase the sum for the integral term
            Pg_ij = self.Pg_stack_ii[node_j, :]  # set the Pg_ij* as a variable to make the code clearer

            if self.leader_acceleration:
                vel_dot_j = calc_derivative(self, node_j) # numerical derivative for neighbors acceleration
                pos_dot_i = vel_i
                vel_dot_i += - np.linalg.inv(K_i)@(Pg_ij @ (self.k_p * (pos_i - pos_j) \
                                                              + self.k_v * (vel_i - vel_j) - vel_dot_j))
                if self.integral_action: # if we want to apply the integral term
                    err_pos_ij = self.error_pos[self.agent_id, node_j, :]
                    pos_dot_i = vel_i
                    vel_dot_i += - np.linalg.inv(K_i)@(Pg_ij@(self.k_p*(pos_i - pos_j) \
                                            + self.k_v*(vel_i - vel_j)+ self.k_i*err_pos_ij - vel_dot_j)) + w_i
            else:
                pos_dot_i = vel_i
                vel_dot_i += - Pg_ij@(self.k_p*(pos_i - pos_j) + self.k_v*(vel_i - vel_j))

                if self.integral_action: # if we want to apply the integral term
                    err_pos_ij = self.error_pos[self.agent_id, node_j, :]
                    pos_dot_i = vel_i
                    vel_dot_i += - (Pg_ij@(self.k_p*(pos_i - pos_j) + self.k_v*(vel_i - vel_j)+ self.k_i*err_pos_ij)) + w_i

            x_dot_i = np.concatenate((pos_dot_i, vel_dot_i))

        x_i += dt * x_dot_i # forward euler to discretize the dynamics

    return x_i

def calc_derivative(self, node_j):
    # function that computes the backward numerical derivative for the neighbor velocity
    n_x = np.shape(self.x_i)[0] # dimension of the state vector
    dd = n_x//2 # dimension of position and velocity vector
    dt = self.communication_time
    index_i = node_j*dd + np.arange(dd) # [0 1], [2 3], ...
    if self.tt == 0:
        vel_dot_j = self.store_acc[index_i, self.tt]/dt
    else:
        vel_j_t = self.store_acc[index_i, self.tt] # velocity of neighbor j at t
        vel_j_t_1 = self.store_acc[index_i, self.tt - 1] # velocity of neighbor j at t-1
        vel_dot_j = (vel_j_t - vel_j_t_1)/dt # acceleration of neighbor j calculated as v_t - v_(t-1) / dt
    return vel_dot_j

def piecewise_acc(self):
    # functions used to define a piecewise continuous acceleration for leaders
    n_x = np.shape(self.x_i)[0]
    dd = n_x//2
    time = np.linspace(0, self.max_iters, self.max_iters + 2)
    tg = quintic(0, 100, time)
    acc = 500*tg.qdd
    acc_t = acc[self.tt]
    acc_t = np.ones(dd)*acc_t
    return acc_t

