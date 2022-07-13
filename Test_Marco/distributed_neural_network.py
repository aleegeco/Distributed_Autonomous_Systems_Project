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
PRINT = True
SAVE = True
# chosen digit to wor
LuckyNumber = 6

epochs = 50
stepsize = 0.005
# stepsize = 1/(k+1)

# Graph parameters
NN = 4
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

J = np.zeros((epochs, NN))

print(f'k\t', end='')
for agent in range(NN):
    if agent == NN-1:
        print(f'J[{agent}]\t d_u[{agent}]\t')
    else:
        print(f'J[{agent}]\t d_u[{agent}]\t', end='\t')

for k in range(epochs-1):
    for agent in range(NN):
        for image in range(dim_train_agent):
            temp_data = data_train[agent, image]
            temp_label = label_train[agent, image]

            xx = forward_pass(uu[k, agent], temp_data, T, dim_layer)

            J_temp, lambdaT = cost_function(xx[-1, 0], temp_label)
            J[k, agent] += J_temp
            temp_delta_u = backward_pass(xx, uu[k, agent], lambdaT, T, dim_layer)
            delta_u[k, agent] += temp_delta_u

    print(f'{k}\t', end='')
    for agent in range(NN):
        uu[k+1, agent] = WW[agent, agent]*uu[k, agent]
        for neigh in G.neighbors(agent):
            uu[k+1, agent] += WW[agent, neigh]*uu[k, neigh]
        uu[k+1, agent] += zz[k, agent] - stepsize*delta_u[k, agent]

        zz[k+1, agent] = WW[agent, agent]*zz[k, agent] - stepsize*(WW[agent, agent]*delta_u[k, agent] - delta_u[k, agent])
        for neigh in G.neighbors(agent):
            zz[k+1, agent] += WW[agent, neigh]*zz[k, neigh] - stepsize*WW[agent, neigh]*delta_u[k, neigh]
        zz[k+1, agent] -= stepsize*delta_u[k, agent]  # not sure that this is needed

        if agent == NN - 1:
            print(f'{J[k, agent]:4.2f}\t {np.linalg.norm(delta_u[k, agent]):4.2f}\t')
        else:
            print(f'{J[k, agent]:4.2f}\t {np.linalg.norm(delta_u[k, agent]):4.2f}\t', end='\t')

if SAVE:
    np.save('store_data/epochs.npy', epochs, allow_pickle=True)
    np.save('store_data/NN.npy', NN, allow_pickle=True)
    np.save('store_data/JJ.npy', J, allow_pickle=True)
    np.save('store_data/delta_u.npy', delta_u, allow_pickle=True)
    np.save('store_data/uu.npy', uu, allow_pickle=True)



_, ax = plt.subplots()
ax.plot(range(epochs), J)
ax.title.set_text('J')
plt.show()

# _, ax = plt.subplots()
# ax.plot(range(epochs), np.linalg.norm(delta_u))
# ax.title.set_text('delta_u')
# plt.show()

val_function(uu, data_test, label_test, dim_test_agent, NN, dim_layer, T)


print('DAJE TUTTO OK')
