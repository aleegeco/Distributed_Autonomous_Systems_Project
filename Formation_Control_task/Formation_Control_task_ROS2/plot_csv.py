
import numpy as np
import matplotlib.pyplot as plt
import signal
import os
import csv
signal.signal(signal.SIGINT, signal.SIG_DFL)


_, _, files = next(os.walk("./_csv_file"))
NN = len(files)

plot = True
animation = True
letters = False
xx_csv = {}
xx_ref_pos_csv = {}
Tlist = []

# load from csv agents dynamics and store them into arrays
for ii in range(NN):
    xx_csv[ii] = np.genfromtxt("_csv_file/agent_{}.csv".format(ii), delimiter=',').T
    Tlist.append(xx_csv[ii].shape[1])

#load from csv agents reference positions and store them into arrays
for ii in range(NN):
    xx_ref_pos_csv[ii] = np.genfromtxt("_csv_file_pos/agent_ref_pos{}.csv".format(ii), delimiter=',').T

# state dimension
n_x = xx_csv[ii].shape[0]
# dimension of position and velocity vectors (i.e. plane X-Y or space X-Y-Z)
dd = n_x//2

T_max = min(Tlist)  # max iterations

# initializations of vectors
xx_pos = np.zeros((NN*dd, T_max))
xx_vel = np.zeros((NN*dd, T_max))
xx_ref_pos = np.zeros((NN*dd, T_max))

# Store differently positions and velocities to plot them separetely
for ii in range(NN):
    index_ii = ii*dd + np.arange(dd)
    xx_pos[index_ii, :] = xx_csv[ii][:dd][:T_max] # useful to remove last samples
    xx_vel[index_ii, :] = xx_csv[ii][dd:][:T_max]
    xx_ref_pos[index_ii, :] = xx_ref_pos_csv[ii][:T_max]

if plot:
# Plot of evolution of position x over time for each node
    legend = []
    plt.figure()
    plt.title("Evolution of $p_{i,x}$")
    for node in range(NN):
        plt.plot(range(T_max), xx_pos[node*2, :])
        legend.append("i: {}".format(node))
    plt.legend(legend); plt.grid()

    # plot of evolution of position y over time for each node
    legend = []
    plt.figure()
    plt.title("Evolution of $p_{i,y}$")
    for node in range(NN):
        plt.plot(range(T_max), xx_pos[node*2 + 1, :])
        legend.append("i: {}".format(node))
    plt.legend(legend); plt.grid()

    # plot of evolution of velocity x over time for each node
    legend = []
    plt.figure()
    plt.title("Evolution of $v_{i,x}$")
    for node in range(NN):
        plt.plot(range(T_max), xx_vel[node*2, :])
        legend.append("i: {}".format(node))
    plt.legend(legend); plt.grid()

    # plot of evolution of velocity y over time for each node
    legend = []
    plt.figure()
    plt.title("Evolution of $v_{i,y}$")
    for node in range(NN):
        plt.plot(range(T_max), xx_vel[node*2 + 1, :])
        legend.append("i: {}".format(node))
    plt.legend(legend); plt.grid()


    # plot error evolution in distance between agents and their reference positions
    legend = []
    plt.figure()
    plt.title("Error evolution $|e_{i,p_{x}}|$")
    for node in range(NN):
        plt.plot(range(T_max), np.abs(xx_pos[node * 2, :] - xx_ref_pos[node * 2]))
        legend.append("i: {}".format(node))
    plt.legend(legend); plt.grid()

    # plot error evolution in distance between agents and their reference positions
    legend = []
    plt.figure()
    plt.title("Error evolution $|e_{i,p_{y}}|$")
    for node in range(NN):
        plt.plot(range(T_max), np.abs(xx_pos[node * 2 + 1, :] - xx_ref_pos[node * 2 + 1]))
        legend.append("i: {}".format(node))
    plt.legend(legend); plt.grid()

# block_var = False if n_x < 3 else True
# plt.show(block=block_var)


# animation of the position of all the agents
if animation: # animation
    plt.figure()
    dt = 10 # sub-sampling of the plot horizon
    for tt in range(0,T_max,dt):
        for ii in range(NN):
            xx_tt = xx_pos[:, tt].T
            xx_ref_tt = xx_ref_pos[:,tt].T
            index_ii =  ii*dd + np.arange(dd)
            xx_ii = xx_tt[index_ii]
            xx_ref_pos_ii = xx_ref_tt[index_ii]
            plt.plot(xx_ii[0],xx_ii[1], marker='o', markersize=15, fillstyle='none', color = 'tab:red')
            plt.plot(xx_ref_pos_ii[0], xx_ref_pos_ii[1], marker='x', markersize=15, color='tab:blue')


        axes_lim = (np.min(xx_pos)-1,np.max(xx_pos)+1)
        plt.xlim(axes_lim); plt.ylim(axes_lim)
        if letters:
            plt.plot(xx_pos[0:dd*NN:dd,:].T,xx_pos[1:dd*NN:dd,:].T, linestyle="dashed")
        plt.grid()
        plt.title("Animation of the simulated formation")
        plt.xlabel("$p_x$"); plt.ylabel("$p_y$")
        plt.axis('equal')

        plt.show(block=False)
        plt.pause(0.1)
        if tt < T_max - dt - 1:
            plt.clf()
    plt.show()
