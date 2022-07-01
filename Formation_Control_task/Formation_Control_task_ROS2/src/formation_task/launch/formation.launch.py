from launch import LaunchDescription
from launch_ros.actions import Node
from formation_task.functions import *
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    MAXITERS = 1000
    COMM_TIME = 10e-2 # communication time period
    NN = 4 # number of agents
    n_leaders = 2 # number of leaders - first two in the vector
    dd = 2 # dimension of position vector and velocity vector
    n_x = 2*dd # dimension of the single vector x_i

    k_p = 0.7 # position gain
    k_v = 1.5 # velocity gain
    k_i = 0.4 # integral gain

    # initialization of the tensor for node reference final positions
    Node_pos = np.zeros((NN, dd, 1))

    # set the position for each agent
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
    xx_init = np.zeros((NN * n_x)).reshape([NN * n_x, 1])

    # tensor which stores all the projection matrix for each pair of nodes (i,j)
    Pg_stack = proj_stack(Node_pos, NN, dd)

    launch_description = [] # Append here your nodes

    # Impose leaders initial conditions - they must start still
    for i in range(n_leaders):
        i_index = n_x*i + np.arange(n_x)
        xx_init[i_index[:dd]] = Node_pos[i,:,:]

    # RVIZ
    # initialization of rviz variables
    rviz_config_dir = get_package_share_directory('formation_task')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    # launch rviz node
    launch_description.append(
        Node(
            package='rviz2',
            node_executable='rviz2',
            arguments=['-d', rviz_config_file],
            ))

    # cycle which create the quantities needed by the source code file and launch the executables needed for the task
    for ii in range(NN):
        N_ii = np.nonzero(Adj[:, ii])[0].tolist()
        ii_index = ii*n_x + np.arange(n_x)
        x_init_ii = xx_init[ii_index].flatten().tolist()
        Pg_stack_ii = Pg_stack[ii, :, :, :].flatten().tolist()

        launch_description.append(
            Node(
                package='formation_task',
                node_namespace='agent_{}'.format(ii),
                node_executable='agent_i',
                parameters=[{
                                'agent_id': ii, 
                                'max_iters': MAXITERS, 
                                'communication_time': COMM_TIME, 
                                'neigh': N_ii, 
                                'xx_init': x_init_ii,
                                'Pg_stack_ii': Pg_stack_ii,
                                'k_p': k_p,
                                'k_v': k_v,
                                'n_leaders': n_leaders,
                                'n_agents': NN,
                                }],
                output='screen',
                prefix='xterm -title "agent_{}" -hold -e'.format(ii)
            ))

        # launch VISUALIZER
        launch_description.append(
            Node(
                package='formation_task',
                node_namespace='agent_{}'.format(ii),
                node_executable='visualizer',
                parameters=[{
                    'agent_id':ii,
                    'communication_time':COMM_TIME,
                    'n_leaders':n_leaders,
                }]
            )
        )

    return LaunchDescription(launch_description)