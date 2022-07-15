from launch import LaunchDescription
from launch_ros.actions import Node
from formation_task.functions import *
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    MAXITERS = 1500
    COMM_TIME = 10e-2 # communication time period
    dd = 2 # dimension of position vector and velocity vector
    n_x = 2*dd # dimension of the single vector x_i

    k_p = 1.5 # position gain
    k_v = 3 # velocity gain
    k_i = 0.3 # integral gain


    acceleration_leader = True # variable which sets the leaders acceleration
    integral_action = False # variable which sets the integral action
    random_init = True # variable which sets randomly the intial conditions

    # dictionary with all the formations we want to try
    formations = {'square': [[1, 5], [5, 5], [1, 1], [5, 1]],
                  'exagon': [[5.75, 5.30], [4.25, 5.30], [6.5, 4.0], [3.50, 4.0], [5.75, 2.70], [4.25, 2.70]],
                  'octagon': [[8, 4], [8, 12], [12, 8], [4, 8], [5.172, 5.172], [10.83, 10.83], [10.83, 5.172], [5.172, 10.83]],
                  'A': [[6, 6], [8, 12], [10, 6], [8, 8], [6.67, 8], [9.33, 8], [7, 9], [9, 9], [7.54, 10.6], [8.47, 10.6]],
                  'B': [[1, 1], [1, 9], [6, 7], [6, 3], [1, 5], [1, 3], [1, 7], [4, 9], [4, 5], [4, 1]],
                  'C': [[7, 1], [7, 9], [6, 1], [6, 9], [5, 2], [5, 8], [4, 3], [4, 7], [4, 4], [4, 6]],
                  'D':[ [8, 6], [8, 14], [12, 10], [8, 10], [8, 8], [8, 12], [10, 13.47], [11.48, 11.97], [11.48, 8], [10, 6.5]],
                  'S':[[8, 10], [7, 6], [9, 14], [6, 12], [10, 8], [6.57, 10.6], [6.57, 13.42], [9.15, 6.36], [9.15, 9.61], [8, 14]],
                  'group': [[2, 8], [6, 4], [8, 8], [4, 6], [8, 2], [5, 9],  [6, 6], [4, 4], [2, 2],  [5, 1]]}


    temp_array = np.array(formations['group'])
    NN = len(temp_array) # number of agents
    n_leaders = NN//2 # number of leaders - firsts positions in the vector

    Node_pos = np.zeros((NN, dd, 1))
    # set the reference position for each agent
    for node in range(NN):
        Node_pos[node, :, :] = temp_array[node, :].reshape((dd, 1))

    # definition of the communication graph and its adjacency matrix
    G = nx.binomial_graph(NN, 1)
    Adj = nx.adjacency_matrix(G).toarray()

    # define initial conditions
    xx_init = np.zeros((NN * n_x)).reshape((NN*n_x,1))

    # tensor which stores all the projection matrix for each pair of nodes (i,j)
    Pg_stack = proj_stack(Node_pos, NN, dd)
    # Compute the Bearing Laplacian to check if the positions and velocities of the followers are singular or not
    B = bearing_laplacian(Pg_stack, Adj, dd)
    Bff = B[dd * n_leaders:dd * NN, dd * n_leaders:dd * NN]

    if np.linalg.det(Bff) != 0:
        print("The matrix Bff is not singular! p*_f and v*_f exist and are unique \n")
    else:
        print("The matrix Bff is singular! Check the Graph \n")

    launch_description = [] # Append here your nodes
    # Impose leaders initial conditions - they must start still
    for i in range(NN):
        i_index = n_x*i + np.arange(n_x)
        if i < n_leaders:
            xx_init[i_index[:dd]] = Node_pos[i,:,:]
        elif random_init:
            xx_init[i_index[:dd]] = 10*np.random.rand(dd).reshape((dd,1))

    # RVIZ
    # initialization of rviz variables
    rviz_config_dir = get_package_share_directory('formation_task')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    # launch rviz node
    launch_description.append(
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config_file],
            ))

    Node_pos = Node_pos.flatten().tolist()
    # cycle which create the quantities needed by the source code ii and launch the executables needed for the task
    for ii in range(NN):
        N_ii = np.nonzero(Adj[:, ii])[0].tolist()
        ii_index = ii*n_x + np.arange(n_x)
        x_init_ii = xx_init[ii_index].flatten().tolist()
        Pg_stack_ii = Pg_stack[ii, :, :, :].flatten().tolist()
        launch_description.append(
            Node(
                package='formation_task',
                namespace='agent_{}'.format(ii),
                executable='agent_i',
                parameters=[{
                                'agent_id': ii, 
                                'max_iters': MAXITERS, 
                                'communication_time': COMM_TIME, 
                                'neigh': N_ii, 
                                'xx_init': x_init_ii,
                                'Pg_stack_ii': Pg_stack_ii,
                                'k_p': k_p,
                                'k_v': k_v,
                                'k_i': k_i,
                                'n_leaders': n_leaders,
                                'n_agents': NN,
                                'node_pos': Node_pos,
                                'leader_acceleration': acceleration_leader,
                                'integral_action': integral_action,
                                }],
                output='screen',
                prefix='xterm -title "agent_{}" -hold -e'.format(ii)
            ))

        # launch VISUALIZER
        launch_description.append(
            Node(
                package='formation_task',
                namespace='agent_{}'.format(ii),
                executable='visualizer',
                parameters=[{
                    'agent_id':ii,
                    'communication_time':COMM_TIME,
                    'n_leaders':n_leaders,
                }]
            )
        )

    return LaunchDescription(launch_description)