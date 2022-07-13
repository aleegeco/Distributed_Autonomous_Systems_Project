import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # this library will be used for data visualization
import networkx as nx  # library for network creation/visualization/manipulation
from Function_Task_1 import *
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from Function_Task_1 import MSE as cost_function
from tabulate import tabulate

np.random.seed(0) # generate random number (always the same seed)


BALANCING = True
FIGURE =False
PRINT = True
# chosen digit to wor
LuckyNumber = 6

epochs = 40
stepsize = 0.001
# stepsize = 1/(k+1)

# Graph parameters
NN = 2
p_ER = 0.4
I_NN = np.identity(NN, dtype=int)  # necessary to build the Adj
dim_train_agent = 200
dim_test_agent = int(0.3*dim_train_agent)

# Data acquisition and processing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# adjusting the type of the data contained in the arrays in this way they can be also negative
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

# scale the brightness of each pixel because otherwise saturates the activation function
x_train = x_train / 255
x_test = x_test / 255

#  Set Up the Neural Network
d = [784, 784, 784, 784]
T = len(d)  # how much layer we have
dim_layer = d[0]  # number of neurons considering bias

while True:
    Adj = np.random.binomial(1, p_ER, (NN, NN))  # create a NN x NN matrix with random connections
    Adj = np.logical_or(Adj, Adj.T)  # makes it symmetric
    Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)  # removes the element on the diagonal

    test = np.linalg.matrix_power((I_NN + Adj), NN)  # necessary condition to check if the graph is connected

    if np.all(test > 0):  # here he tests if the matrix that he created is connected
        if PRINT:
            print("Congratulation, the graph is connected.")
        break
    else:
        if PRINT:
            print("Warning, the graph is NOT connected.")
        quit()

#  Compute mixing matrices
#  Metropolis-Hastings method to obtain a doubly-stochastic matrix

WW = np.zeros((NN, NN))

for ii in range(NN):
    N_ii = np.nonzero(Adj[ii])[0]  # In-Neighbors of node i
    deg_ii = len(N_ii)

    for jj in N_ii:
        N_jj = np.nonzero(Adj[jj])[0]  # In-Neighbors of node j
        deg_jj = N_jj.shape[0]
        WW[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

WW += I_NN - np.diag(np.sum(WW, axis=0))

if PRINT:
    print('Check Stochasticity \n row: {} \n column: {}'.format(np.sum(WW, axis=1), np.sum(WW, axis=0)))

# Creating the Graph g
G = nx.from_numpy_array(Adj)
if FIGURE:
    nx.draw(G, with_labels=True, font_weight='bold', node_color='orange', node_size=800)

# we associate -1 (or 0) to data which not represent the number we want to classify
for i in range(0, np.shape(y_train)[0]):
    if y_train[i] == LuckyNumber:
        y_train[i] = 1
    else:
        y_train[i] = 0

for i in range(0, np.shape(y_test)[0]):
    if y_test[i] == LuckyNumber:
        y_test[i] = 1
    else:
        y_test[i] = 0

# Reshape of the input data from a matrix [28 x 28] to a vector [ 784 x 1 ]
x_train_vct = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test_vct = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

# Under-sampling to make the dataset balanced
if BALANCING:
    rus = RandomUnderSampler()
    x_train_vct, y_train = rus.fit_resample(x_train_vct, y_train)
    x_test_vct, y_test = rus.fit_resample(x_test_vct, y_test)

    x_train_vct, _, y_train, _ = train_test_split(x_train_vct, y_train, test_size=0.01)
    x_test_vct, _, y_test, _ = train_test_split(x_test_vct, y_test, test_size=0.01)

    print('Resampled dataset shape %s' % Counter(y_train))

data_train = np.zeros((NN, dim_train_agent, np.shape(x_train_vct)[1]))
label_train = np.zeros((NN, dim_train_agent))

data_test = np.zeros((NN, dim_test_agent, np.shape(x_test_vct)[1]))
label_test = np.zeros((NN, dim_test_agent))

# data_validation, label_validation
for agent in range(NN):
    agent_index = agent * dim_train_agent + np.arange(dim_train_agent)
    data_train[agent, :, :] = x_train_vct[agent_index, :]
    label_train[agent, :] = y_train[agent_index]

    agent_index = agent * dim_test_agent + np.arange(dim_test_agent)
    data_test[agent, :, :] = x_test_vct[agent_index, :]
    label_test[agent, :] = y_test[agent_index]

# Initialization
uu = np.zeros((epochs, NN, T-1, dim_layer, dim_layer+1))
uu[0] = np.random.randn(NN, T-1, dim_layer, dim_layer+1)
uu[:, :, -1, 1:] = 0

zz = np.zeros_like(uu)
delta_u = np.zeros_like(uu)

#delta_u_norm = np.zeros((epochs, NN))

J = np.zeros((epochs, NN))

print('k \t J[0]\t Delta_U[0]\t J[1]\t Delta_U[1]\t ')
print('---------------------------------------------------------')

for k in range(epochs-1):
    for agent in range(NN):
        #Delta_u = 0
        for image in range(dim_train_agent):
            temp_data = data_train[agent, image]
            temp_label = label_train[agent, image]

            xx = forward_pass(uu[k, agent], temp_data, T, dim_layer)

            J_temp, lambdaT = cost_function(xx[-1, 0], temp_label)
            J[k, agent] += J_temp
            temp_delta_u = backward_pass(xx, uu[k, agent], lambdaT, T, dim_layer)
            delta_u[k, agent] += temp_delta_u

        #delta_u_norm[k, agent] = np.linalg.norm(Delta_u)

    for agent in range(NN):
        uu[k+1, agent] = WW[agent, agent]*uu[k, agent] + zz[k, agent] - stepsize*delta_u[k, agent]
        for neigh in G.neighbors(agent):
            uu[k+1, agent] += WW[agent, neigh]*uu[k, neigh]

        zz[k+1, agent] = WW[agent, agent]*zz[k, agent] - stepsize*(WW[agent, agent]*delta_u[k, agent] - delta_u[k, agent])
        for neigh in G.neighbors(agent):
            zz[k+1] += WW[agent, neigh]*zz[k, neigh] - stepsize*(WW[agent, neigh]*delta_u[k, neigh] - delta_u[k, agent])

    print(f'{k}\t {J[k, 0]:4.2f}\t {np.linalg.norm(delta_u[k, 0]):4.2f}\t {J[k, 1]:4.2f}\t {np.linalg.norm(delta_u[k, 1]):4.2f}\t')



_, ax = plt.subplots()
ax.semilogy(range(epochs), J)
ax.title.set_text('J')
plt.show()

_, ax = plt.subplots()
ax.semilogy(range(epochs), delta_u_norm)
ax.title.set_text('delta_u')
plt.show()

counter_corr_label = 0
correct_predict = 0
correct_predict_not_lucky = 0
false_positive = 0
false_negative = 0
agent = 0
for image in range(dim_test_agent):
    xx = forward_pass(uu[-1, agent], data_test[agent, image], T, dim_layer)
    predict = xx[-1, 0]

    if y_test[image] == 1:
        counter_corr_label += 1
    if (predict >= 0.5) and (label_test[agent, image] == 1):
        correct_predict += 1
    elif (predict < 0.5) and (label_test[agent, image] == 0):
        correct_predict_not_lucky += 1
    elif (predict < 0.5) and (label_test[agent, image] == 1):
        false_negative += 1
    elif (predict >= 0.5) and (label_test[agent,image] == 0):
        false_positive += 1

print("The accuracy is {} % where:\n".format((
                                                         correct_predict + correct_predict_not_lucky) / dim_test_agent * 100))  # sum of first and second category expressed in percentage
print("\tFalse positives {} \n".format(false_positive))  # third category ( false positive)
print("\tFalse negatives {} \n".format(false_negative))  # fourth category ( false negative)
print("\tNumber of times LuckyNumber has been identified correctly {} over {} \n".format(correct_predict,
                                                                                         dim_test_agent))  # first category ( images associated to lable 1 predicted correctly )
print("\tNumber of times not LuckyNumber has been identified correctly {} over {} \n".format(correct_predict_not_lucky,
                                                                                             dim_test_agent))  # first category ( images associated to lable 1 predicted correctly )
print("The effective LuckyNumbers in the tests are: {}".format(counter_corr_label))
