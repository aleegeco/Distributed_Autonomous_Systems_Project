from launch import LaunchDescription
from launch_ros.actions import Node
from formation_task_letters.functions_letters import *
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    MAXITERS = 1000
    COMM_TIME = 10e-1 # communication time period
    NN = 10 # number of agents
    n_leaders = 4 # number of leaders - first two in the vector
    dd = 2 # dimension of position vector and velocity vector
    n_x = 2*dd # dimension of the single vector x_i

    k_p = 1 # position gain
    k_v = 1.5 # velocity gain
    k_i = 0.4 # integral gain

    integral_action = False

    formations = np.zeros((3,NN,dd))
    # letter A
    formations[0, :, :] = np.array([[1, 1], [9, 1], [5, 9], [5, 5], [2, 3], [3, 5], [4, 7], [6, 7], [7, 5], [8, 3]]).reshape([NN,dd])
    # letter B
    formations[1, :, :] = np.array([[1, 1], [1, 9], [6, 7], [6, 3], [1, 5], [1, 3], [1, 7], [4, 9], [4, 5], [4, 1]]).reshape([NN,dd])
    # letter C
    formations[2, :, :] = np.array([[7, 1], [7, 9], [6, 1], [6, 9], [5, 2], [5, 8], [4, 3], [4, 7], [4, 4], [4, 6]]).reshape([NN,dd])

    Pg_stack = proj_stack(formations[0,:,:], NN, dd)
    # definition of the communication graph and its adjacency matrix
    G = nx.binomial_graph(NN, 0.9)
    Adj = nx.adjacency_matrix(G).toarray()

    # define initial positions
    xx_init = np.zeros((NN*n_x)).reshape([NN*n_x, 1])

    for node in range(n_leaders):
        i_index = node*n_x + np.arange(n_x)
        xx_init[i_index[:dd],:] = formations[0, node, :].reshape((dd,1))

    launch_description = [] # Append here your nodes

    # RVIZ
    # initialization of rviz variables
    rviz_config_dir = get_package_share_directory('formation_task_letters')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    # launch rviz node
    launch_description.append(
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config_file],
            ))

    formations = formations.flatten().tolist()
    # cycle which create the quantities needed by the source code file and launch the executables needed for the task
    for ii in range(NN):
        N_ii = np.nonzero(Adj[:, ii])[0].tolist()
        ii_index = ii*n_x + np.arange(n_x)
        x_init_ii = xx_init[ii_index].flatten().tolist()
        Pg_stack_ii = Pg_stack[ii, :, :].flatten().tolist()
        launch_description.append(
            Node(
                package='formation_task_letters',
                namespace='agent_{}'.format(ii),
                executable='agent_i',
                parameters=[{
                                'agent_id': ii, 
                                'max_iters': MAXITERS, 
                                'communication_time': COMM_TIME, 
                                'neigh': N_ii, 
                                'xx_init': x_init_ii,
                                'k_p': k_p,
                                'k_v': k_v,
                                'k_i': k_i,
                                'integral_action': integral_action,
                                'n_leaders': n_leaders,
                                'n_agents': NN,
                                'formation': formations,
                                'proj_stack': Pg_stack_ii,
                                }],
                output='screen',
                prefix='xterm -title "agent_{}" -hold -e'.format(ii)
            ))

        # launch VISUALIZER
        launch_description.append(
            Node(
                package='formation_task_letters',
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