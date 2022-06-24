from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx
from Formation_Control_task.functions import *

def generate_launch_description():
    MAXITERS = 200
    COMM_TIME = 5e-2 # communication time period
    NN = 4 # number of agents
    n_leaders = 2 # number of leaders
    dd = 2 # dimension of position vector and velocity vector
    n_x = 2*dd # dimension of the single vector x_i

    k_p = 0.7 # position gain
    k_v = 1.5 # velocity gain

    Node_pos = np.zeros((NN, dd, 1))

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

    # definition of the communication graph and its adjacency matrix
    G = nx.binomial_graph(NN, 0.9)
    Adj = nx.adjacency_matrix(G).toarray()

    # definite initial positions
    xx_init = np.zeros((NN * n_x, 1))


    Pg_stack = proj_stack(Node_pos, NN, dd)

    launch_description = [] # Append here your nodes

    # Impose leaders initial conditions - they must start still
    for i in range(n_leaders):
        for j in range(dd):
            xx_init[(n_leaders * i) + j] = Node_pos[i, j, :]

    # cycle which create the quantities needed by the source code file
    for ii in range(NN):
        pos_ii = Node_pos[ii,:,:].tolist()
        N_ii = np.nonzero(Adj[:, ii])[0].tolist()
        ii_index = ii*n_x + np.arange(n_x)
        x_init_ii = xx_init[ii_index].flatten().tolist()
        Pg_stack_ii = Pg_stack[ii, :, :, :].flatten().tolist()

        launch_description.append(
            Node(
                package='formation_task',
                node_namespace ='agent_{}'.format(ii),
                node_executable='agent_i',
                parameters=[{
                                'agent_id': ii, 
                                'max_iters': MAXITERS, 
                                'communication_time': COMM_TIME, 
                                'neigh': N_ii, 
                                'xx_init': x_init_ii,
                                'pos_xy' : pos_ii,
                                'Pg_stack_ii': Pg_stack_ii,
                                'k_p': k_p,
                                'k_v': k_v,
                                }],
                output='screen',
                prefix='xterm -title "agent_{}" -hold -e'.format(ii)
            ))

    return LaunchDescription(launch_description)