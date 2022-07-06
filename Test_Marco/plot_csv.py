import numpy as np
import matplotlib.pyplot as plt  # this library will be used for data visualizationNN = 4



max_iters = np.load('store_data/max_iter.npy')
d = [784, 784, 784, 784]
T = len(d)  # how much layer we have
dim_layer = d[0]  # number of neurons considering bias
NN = np.load('store_data/NN.npy')


JJ = np.load('store_data/JJ.npy')
print('JJ downloaded')
dJJ_norm = np.load('store_data/dJJ.npy')
print('dJJ downloaded')
uu = np.load('store_data/uu.npy')
print('uu downloaded')
yy = np.load('store_data/yy.npy')
print('yy downloaded')

plt.figure()
for agent in range(NN):
    plt.semilogy(range(max_iters-1), (JJ[agent, :-1]))

plt.figure()
for agent in range(NN):
    plt.plot(range(max_iters-1), dJJ_norm[agent, :-1])

nn = 3
fig, axs = plt.subplots(nn, nn)

for i in range(nn):
    for j in range(nn):

        random_node = int(np.random.rand()*(d[1]))
        random_weight = int(np.random.rand()*(d[1]))

        for agent in range(NN):
            axs[i,j].plot(range(max_iters-1), uu[agent, :-1, 2 , random_node, random_weight])


print('DAJE TUTTO OK')
