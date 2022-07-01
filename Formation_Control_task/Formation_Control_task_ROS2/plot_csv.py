
import numpy as np
import matplotlib.pyplot as plt
import signal
import os
import csv
signal.signal(signal.SIGINT, signal.SIG_DFL)


_, _, files = next(os.walk("./_csv_file"))
NN = len(files)

animation = False
xx_csv = {}
Tlist = []

for ii in range(NN):
    xx_csv[ii] = np.genfromtxt("_csv_file/agent_{}.csv".format(ii), delimiter=',').T
    Tlist.append(xx_csv[ii].shape[1])

xx_ref_pos_csv = {}
for ii in range(NN):
    xx_ref_pos_csv[ii] = np.genfromtxt("_csv_file_pos/agent_ref_pos{}.csv".format(ii), delimiter=',').T

n_x = xx_csv[ii].shape[0]
dd = n_x//2

print(n_x)
T_max = min(Tlist)

xx_pos = np.zeros((NN*dd, T_max))
xx_vel = np.zeros((NN*dd, T_max))
xx_ref_pos = np.zeros((NN*dd))

# Store differently positions and velocities
for ii in range(NN):
    index_ii = ii*dd + np.arange(dd)
    xx_pos[index_ii, :] = xx_csv[ii][:dd][:T_max] # useful to remove last samples
    xx_vel[index_ii, :] = xx_csv[ii][dd:][:T_max]
    xx_ref_pos[index_ii] = xx_ref_pos_csv[ii][:]


# Plot of evolution of position x over time for each node
legend = []
plt.figure()
plt.title("Evolution of $p_{i,x}$")
for node in range(NN):
    print(node)
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

legend = []
plt.figure()
plt.title("Error evolution $|e_{i,p_{x}}|$")
for node in range(NN):
    plt.plot(range(T_max), np.abs(xx_pos[node*2, :] - xx_ref_pos[node*2]))
    legend.append("i: {}".format(node))
plt.legend(legend); plt.grid()

legend = []
plt.figure()
plt.title("Error evolution $|e_{i,p_{y}}|$")
for node in range(NN):
    plt.plot(range(T_max), np.abs(xx_pos[node*2 + 1, :] - xx_ref_pos[node*2 + 1]))
    legend.append("i: {}".format(node))
plt.legend(legend); plt.grid()


# block_var = False if n_x < 3 else True
# plt.show(block=block_var)
#

if animation: # animation
    plt.figure()
    dt = 3 # sub-sampling of the plot horizon
    for tt in range(0,T_max,dt):
        xx_tt = xx_pos[:,tt].T
        for ii in range(NN):
            index_ii =  ii*dd + np.arange(dd)
            xx_ii = xx_tt[index_ii]
            plt.plot(xx_ii[0],xx_ii[1], marker='o', markersize=15, fillstyle='none', color = 'tab:red')


        axes_lim = (np.min(xx_pos)-1,np.max(xx_pos)+1)
        plt.xlim(axes_lim); plt.ylim(axes_lim)
        plt.plot(xx_pos[0:dd*NN:dd,:].T,xx_pos[1:dd*NN:dd,:].T)

        plt.axis('equal')

        plt.show(block=False)
        plt.pause(0.1)
        if tt < T_max - dt - 1:
            plt.clf()
    plt.show()
