import numpy as np
import matplotlib.pyplot as plt  # this library will be used for data visualizationNN = 4
from Function_Task_1 import *

max_iters = np.load('store_data/max_iter.npy')
d = [784, 784, 784]
T = len(d)  # how much layer we have
dim_layer = d[0]  # number of neurons considering bias
NN = np.load('store_data/NN.npy')

JJ = np.load('store_data/JJ.npy')
print('JJ downloaded')
delta_u = np.load('store_data/delta_u.npy')
print('delta_u_store downloaded')
uu = np.load('store_data/uu.npy')
print('uu downloaded')
yy = np.load('store_data/yy.npy')
print('yy downloaded')

plt.figure()
for agent in range(NN):
    plt.semilogy(range(max_iters-1), (JJ[agent, :-1]))
plt.title("Cost Function"); plt.grid()
plt.xlabel("iteration")
plt.ylabel("J")

norm = np.zeros(max_iters)
for i in range(max_iters):
    norm[i] = np.linalg.norm(delta_u[1, i])



plt.figure()
for agent in range(NN):
    plt.semilogy(range(max_iters-1), (norm[:-1]))
plt.title("Gradient of the cost function"); plt.grid()
plt.xlabel("iteration")
plt.ylabel("dJ")


nn = 2
fig, axs = plt.subplots(nn, nn)
for i in range(nn):
    for j in range(nn):
        random_node = int(np.random.rand()*(d[1]))
        random_weight = int(np.random.rand()*(d[1]))
        for agent in range(NN):
            axs[i,j].plot(range(max_iters-1), uu[agent, :-1, 0, random_node, random_weight])
            axs[i,j].margins(0)
plt.show()

y_test = np.load("store_data/y_test.npy")
x_test_vct = np.load("store_data/x_test_vct.npy")
val_function(uu[0,-1], x_test_vct,y_test, T, dim_layer, np.shape(y_test)[0])
print('DAJE TUTTO OK')
