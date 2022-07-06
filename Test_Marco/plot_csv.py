import numpy as np
import matplotlib.pyplot as plt  # this library will be used for data visualizationNN = 4



max_iters = np.load('max_iter.npy')
d = [784, 784, 784, 784]
T = len(d)  # how much layer we have
dim_layer = d[0]  # number of neurons considering bias
NN = np.load('NN.npy')


JJ = np.load('JJ.npy')
print('JJ downloaded')
dJJ_norm = np.load('dJJ.npy')
print('dJJ downloaded')
uu = np.load('uu.npy')
print('uu downloaded')
yy = np.load('yy.npy')
print('yy downloaded')
grad_u = np.load('grad_u.npy')
print('grad_u downloaded')

plt.figure()
for agent in range(NN):
    plt.semilogy(range(max_iters-1), (JJ[agent, :-1]))

plt.figure()
for agent in range(NN):
    plt.plot(range(max_iters-1), dJJ_norm[agent, :-1])

# plt.figure()
# plt.plot(range(max_iters), grad_u[0, :, -2, 1, :])
# plt.show()

print('DAJE TUTTO OK')
