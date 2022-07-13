import numpy as np
import matplotlib.pyplot as plt  # this library will be used for data visualizationNN = 4
from Function_Task_1 import *

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
for agent in range(NN):
    plt.semilogy(range(epochs-1), (JJ[:-1, agent]))
plt.title("Cost Function"); plt.grid()
plt.xlabel("iteration")
plt.ylabel("J")


plt.figure()

for agent in range(NN):
    temp_norm = np.zeros(epochs)
    for k in range(epochs):
        temp_norm[k] = np.linalg.norm(delta_u[k, agent])
    plt.plot(range(epochs-1), temp_norm[:-1])

nn = 2
fig, axs = plt.subplots(nn, nn)
for i in range(nn):
    for j in range(nn):
        random_node = int(np.random.rand()*(d[1]))
        random_weight = int(np.random.rand()*(d[1]))
        for agent in range(NN):
            axs[i, j].plot(range(epochs-1), uu[:-1, agent, 1, random_node, random_weight])
            axs[i, j].margins(0)
plt.show()

print('DAJE TUTTO OK')
