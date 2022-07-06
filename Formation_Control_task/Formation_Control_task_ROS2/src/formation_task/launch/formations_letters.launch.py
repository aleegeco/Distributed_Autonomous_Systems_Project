from launch import LaunchDescription
from launch_ros.actions import Node
from formation_task.functions_lett import *
import numpy as np
import networkx as nx
import os
from ament_index_python.packages import get_package_share_directory
from roboticstoolbox.tools.trajectory import *



def generate_launch_description():
    COMM_TIME = 10e-2 # communication time period
    dd = 2 # dimension of position vector and velocity vector
    n_x = 2*dd # dimension of the single vector x_i

    NN = 10
    n_leaders = NN//2 - 1

    k_p = 1.5 # position gain
    k_v = 3 # velocity gain
    k_i = 0.3 # integral gain


    acceleration_leader = True # variable which sets the leaders acceleration
    integral_action = False # variable which sets the integral action
    random_init = True # variable which sets randomly the intial conditions

    # definition of the communication graph and its adjacency matrix
    G = nx.binomial_graph(NN, 1)
    Adj = nx.adjacency_matrix(G).toarray()

    # dictionary with all the formations we want to try
    formations = {'A': [[6, 6], [8, 12], [10, 6], [8, 8], [6.67, 8], [9.33, 8], [7, 9], [9, 9], [7.54, 10.6], [8.47, 10.6]],
                  'B': [[1, 1], [1, 9], [6, 7], [6, 3], [1, 5], [1, 3], [1, 7], [4, 9], [4, 5], [4, 1]],
                  'C': [[7, 1], [7, 9], [6, 1], [6, 9], [5, 2], [5, 8], [4, 3], [4, 7], [4, 4], [4, 6]],
                  'D':[ [8, 6], [8, 14], [12, 10], [8, 10], [8, 8], [8, 12], [10, 13.47], [11.48, 11.97], [11.48, 8], [10, 6.5]],
                  'S':[[8, 10], [7, 6], [9, 14], [6, 12], [10, 8], [6.57, 10.6], [6.57, 13.42], [9.15, 6.36], [9.15, 9.61], [8, 14]]}

    word = 'DAS'
    n_letters = len(word)

    step = 1000
    MAX_ITERS = step*n_letters
    transient = 400


    Pg_stack_word = np.zeros((n_letters, NN, NN, dd, dd))
    Node_pos = np.zeros((n_letters, NN, dd, 1))
    # def tensor for the node pos
    for letter in range(n_letters):
        let = word[letter]
        temp_array = np.array(formations[let])
        for node in range(NN):
            Node_pos[letter, node, :, :] = temp_array[node, :].reshape((dd, 1))

        Pg_stack_word[letter, :, :, :, :] = proj_stack(Node_pos[letter, :, :, :], NN, dd)


    launch_description = [] # Append here your nodes

    # define initial conditions
    xx_init = np.zeros((NN * n_x)).reshape((NN*n_x,1))
    # Impose leaders initial conditions - they must start still
    if random_init:
        for i in range(NN):
            i_index = n_x*i + np.arange(n_x)
            xx_init[i_index[:dd]] = 10*np.random.rand(dd).reshape((dd,1))



    acc_profile_store = acc_profile(n_leaders, n_letters, dd, n_x, MAX_ITERS, transient, step, Node_pos, xx_init).flatten().tolist()

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
        Pg_stack_word_ii = Pg_stack_word[:,ii, :, :, :].flatten().tolist()
        launch_description.append(
            Node(
                package='formation_task',
                namespace='agent_{}'.format(ii),
                executable='agent_lett_i',
                parameters=[{
                    'agent_id': ii,
                    'max_iters': MAX_ITERS,
                    'communication_time': COMM_TIME,
                    'neigh': N_ii,
                    'xx_init': x_init_ii,
                    'Pg_stack_word_ii': Pg_stack_word_ii,
                    'k_p': k_p,
                    'k_v': k_v,
                    'k_i': k_i,
                    'n_leaders': n_leaders,
                    'n_agents': NN,
                    'n_letters': n_letters,
                    'node_pos': Node_pos,
                    'leader_acceleration': acceleration_leader,
                    'integral_action': integral_action,
                    'acceleration_profile' : acc_profile_store
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