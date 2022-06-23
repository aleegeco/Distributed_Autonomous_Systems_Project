from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import networkx as nx

def generate_launch_description():
    MAXITERS = 200
    COMM_TIME = 5e-2 # communication time period
    NN = 6 # number of agents
    n_x = 2 # dimension of x_i 

    # Weight matrix to control inter-agent distances
    L = 2
    D = 2*L
    H = np.sqrt(3)*L

    # minimally rigid 2*N-3 (only for regular polygons)
    # rigid
    distances = [[0,     L,      0,    D,     H,    L],
                [L,     0,      L,    0,     D,    0],
                [0,     L,      0,    L,     0,    D],     
                [D,     0,      L,    0,     L,    0],     
                [H,     D,      0,    L,     0,    L],     
                [L,     0,      D,    0,     L,    0]]

    distances = np.asarray(distances) #convert list to numpy array

    # Adjacency matrix
    Adj = distances > 0

    # definite initial positions
    x_init = np.random.rand(n_x*NN,1)
    
    launch_description = [] # Append here your nodes
    
    for ii in range(NN):
        distances_ii = distances[:, ii].tolist()

        N_ii = np.nonzero(Adj[:, ii])[0].tolist()
        ii_index = ii*n_x + np.arange(n_x)
        x_init_ii = x_init[ii_index].flatten().tolist()

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
                                'x_init': x_init_ii,
                                'dist' : distances_ii,
                                }],
                output='screen',
                prefix='xterm -title "agent_{}" -hold -e'.format(ii)
            ))

    return LaunchDescription(launch_description)