import numpy as np
import matplotlib.pyplot as plt  # this library will be used for data visualizationNN = 4



max_iters = 30
d = [784, 784, 784, 784]
T = len(d)  # how much layer we have
dim_layer = d[0]  # number of neurons considering bias
NN =

JJ = np.loadtxt('JJ.csv', delimiter='::').reshape((NN, max_iters))
dJJ_norm = np.loadtxt('dJJ_norm.csv', delimiter='::').reshape((NN, max_iters))
uu = np.loadtxt('uu.csv', delimiter='::').reshape((NN, max_iters, T - 1, dim_layer, dim_layer + 1))
yy = np.loadtxt('yy.csv', delimiter='::').reshape((NN, max_iters, T - 1, dim_layer, dim_layer + 1))
grad_u = np.loadtxt('grad_u.csv', delimiter='::').reshape((NN, max_iters, T - 1, dim_layer, dim_layer + 1))


plt.figure()
plt.plot(range(max_iters-1), (JJ[0,:-1]))

plt.figure()
plt.plot(range(max_iters), dJJ_norm[0,:])

plt.figure()
plt.plot(range(max_iters), grad_u[0, :, -2, 1, :])