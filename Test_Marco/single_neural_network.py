import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # this library will be used for data visualization
import networkx as nx  # library for network creation/visualization/manipulation
from Function_Task_1 import *
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from Function_Task_1 import MSE as cost_function

np.random.seed(0) # generate random number (always the same seed)


BALANCING = True
FIGURE =False

# chosen digit to wor
LuckyNumber = 6

epochs = 50
stepsize = 0.01
# stepsize = 1/(k+1)
NN = 2
dim_train_agent = 200
dim_test_agent = int(0.2*dim_train_agent)

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

    x_train_vct, _, y_train, _ = train_test_split(x_train_vct, y_train, test_size=0.01, shuffle=True)
    x_test_vct, _, y_test, _ = train_test_split(x_test_vct, y_test, test_size=0.01, shuffle=True)

    print('Resampled dataset shape %s' % Counter(y_train))

    if FIGURE:
        plt.figure()
        for i in range(100):
            plt.subplot(10, 10, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(np.reshape(x_train_vct[i], (28, 28)))
            plt.xlabel(y_train[i])
        plt.show()

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

uu = np.random.randn(epochs, NN, T-1, dim_layer, dim_layer+1)
uu[:, :, -1, 1:] = 0
delta_u_store = np.zeros((epochs, NN))
Delta_u = 0
J = np.zeros((epochs, NN))

for k in range(epochs-1):
    for agent in range(NN):
        Delta_u = 0
        for image in range(dim_train_agent):
            temp_data = data_train[agent, image]
            temp_label = label_train[agent, image]

            xx = forward_pass(uu[k, agent], temp_data, T, dim_layer)

            J_temp, lambdaT = cost_function(xx[-1, 0], temp_label)
            J[k, agent] += J_temp
            Delta_u += backward_pass(xx, uu[k, agent], lambdaT, T, dim_layer)

        delta_u_store[k, agent] = np.linalg.norm(Delta_u)
        uu[k+1, agent] = uu[k, agent] - stepsize * Delta_u
        print(f'agent: {agent}, Iteration: {k}, Loss function: {J[k, agent]:4.3f}, Delta_u: {delta_u_store[k, agent]:4.3f}')


_, ax = plt.subplots()
ax.plot(range(epochs), J)
ax.title.set_text('$J$')
# ax.xlabel("iterations")
# ax.ylabel("$J(.)$")
plt.show()

_, ax = plt.subplots()
ax.plot(range(epochs), delta_u_store)
ax.title.set_text('$\Delta u$')
# ax.xlabel("iterations")
# ax.ylabel("$\Delta u$")
plt.show()

val_function(uu, data_test, label_test, dim_test_agent, NN, dim_layer, T)