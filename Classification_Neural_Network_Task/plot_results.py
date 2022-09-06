import numpy as np
import matplotlib.pyplot as plt  # this library will be used for data visualizationNN = 4
from Function_Task_1 import *

# Load Parameters from store folder after the
epochs = np.load('store_data/epochs.npy')
d = [784, 784, 784, 784]
T = len(d)  # how much layer we have
dim_layer = d[0]  # number of neurons considering bias
NN = np.load('store_data/NN.npy')
JJ = np.load('store_data/JJ.npy')
print('JJ downloaded')
delta_u = np.load('store_data/delta_u.npy')
print('delta_u downloaded')
uu = np.load('store_data/uu.npy')
print('uu downloaded')


plt.figure()
legend = []
for agent in range(NN):
    legend.append("i = {}".format(agent))
    plt.plot(range(epochs-1), (JJ[:-1, agent]))
plt.title("Agents Cost Functions $\mathcal{J}_i$")
plt.grid()
plt.xlabel("iteration")
plt.ylabel("$\mathcal{J}_i$")
plt.legend(legend)

plt.figure()
temp_sum_JJ = np.zeros(epochs)
for k in range(epochs):
    temp_sum_JJ[k] = np.sum(JJ[k])
plt.plot(range(epochs-1), temp_sum_JJ[:-1])
plt.title("Sum of the agents cost functions: $\sum^{N}_{i=0}\ \mathcal{J}_i$")
plt.grid()
plt.xlabel("iteration")
plt.ylabel("$\sum_i\ \mathcal{J}_i$")

plt.figure()
legend = []
for agent in range(NN):
    legend.append("i = {}".format(agent))
    temp_norm = np.zeros(epochs)
    for k in range(epochs):
        temp_norm[k] = np.linalg.norm(delta_u[k, agent])
    plt.plot(range(epochs-1), temp_norm[:-1])
plt.title("Norm of $\Delta u$ for each agent")
plt.grid()
plt.legend(legend)
plt.xlabel("iteration")
plt.ylabel("$||\Delta u_i||$")

plt.figure()
temp_sum_norm = np.zeros(epochs)
for k in range(epochs):
    temp_sum_norm[k] = np.sum(np.linalg.norm(delta_u[k]))
plt.plot(range(epochs-1), temp_sum_norm[:-1])
plt.title("Sum of the $\Delta u$ norm of each agent: $\sum^{N}_{i=0}\ ||\Delta u_i||$")
plt.grid()
plt.xlabel("iteration")
plt.ylabel("$\sum_i\ ||\Delta u_i||$")


nn = 2
legend = []
fig, axs = plt.subplots(nn, nn)
for i in range(nn):
    for j in range(nn):
        random_node = int(np.random.rand()*(d[1]))
        random_weight = int(np.random.rand()*(d[1]))
        for agent in range(NN):
            axs[i, j].plot(range(epochs - 1), uu[:-1, agent, 1, random_node, random_weight])
            axs[i, j].margins(0)
            axs[i, j].grid(True)
            axs[i, j].set_title(f'Neuron:{random_node}, Weight:{random_weight}, layer:{1}')
plt.setp(axs[-1, :], xlabel='iteration')
plt.setp(axs[:, 0], ylabel='weight')
fig.suptitle("Consensus over weights")
plt.show()

print('DAJE TUTTO OK')
