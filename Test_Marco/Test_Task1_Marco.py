from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # this library will be used for data visualization
import networkx as nx  # library for network creation/visualization/manipulation
from Function_Task_1 import *
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
# np.random.seed(0)  # generate random number (always the same seed)

PRINT = True
FIGURE = False
RESIZE_DATA = False
SAVE = True
BALANCING = False

# Parameters for the data size
test_set_size = 0.1  # percentage of the test set over the entire data
percent = 0.1  # percentage of data we want to give to our system from all the data available

# chosen digit to work on
LuckyNumber = 4

# DEFINITION OF THE BINOMIAL GRAPH
NN = 4  # number of AGENTS
p_ER = 0.3  # spawn edge probability
I_NN = np.identity(NN, dtype=int)  # necessary to build the Adj

# Main ALGORITHM Parameters
max_iters = 20
stepsize = 0.01

dim_train_agent = 100  # impose the number of images
dim_test_agent = 100

while 1:
    Adj = np.random.binomial(1, p_ER, (NN, NN))  # create a NN x NN matrix with random connections
    Adj = np.logical_or(Adj, Adj.T)  # makes it symmetric
    Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)  # removes the element on the diagonal

    test = np.linalg.matrix_power((I_NN + Adj), NN)  # necessary condition to check if the graph is connected

    if np.all(test > 0):  # here he tests if the matrix that he created is connected
        if PRINT: print("Congratulation, the graph is connected.")
        break
    else:
        if PRINT: print("Warning, the graph is NOT connected.")
        quit()

#  Compute mixing matrices
#  Metropolis-Hastings method to obtain a doubly-stochastic matrix

WW = np.zeros((NN, NN))

for ii in range(NN):
    N_ii = np.nonzero(Adj[ii])[0]  # In-Neighbors of node i
    deg_ii = len(N_ii)

    for jj in N_ii:
        N_jj = np.nonzero(Adj[jj])[0]  # In-Neighbors of node j
        # deg_jj = len(N_jj)
        deg_jj = N_jj.shape[0]

        WW[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))
        # WW[ii,jj] = 1/(1+np.max(np.stack((deg_ii,deg_jj)) ))

WW += I_NN - np.diag(np.sum(WW, axis=0))

if PRINT: print('Check Stochasticity \n row: {} \n column: {}'.format(np.sum(WW, axis=1), np.sum(WW, axis=0)))

# Creating the Graph g
G = nx.from_numpy_array(Adj)
if FIGURE: nx.draw(G, with_labels=True, font_weight='bold', node_color='orange', node_size=800)

# Data acquisition and processing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# adjusting the type of the data contained in the arrays in this way they can be also negative
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

# scale the brightness of each pixel because otherwise saturates the activation function
x_train = x_train / 255
x_test = x_test / 255

# Reducing the datas if required
if RESIZE_DATA:
    x_total_temp = np.append(x_train, x_test, axis=0)
    x_total = x_total_temp[0: int(np.shape(x_total_temp)[0] * percent)]
    y_total_temp = np.append(y_train, y_test, axis=0)
    y_total = y_total_temp[0: int(np.shape(y_total_temp)[0] * percent)]

    # Random redistribution of the data in two sets ( test and train)
    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=test_set_size)

# we associate -1 to data which not represent the number we want to calssify
for i in range(0, np.shape(y_train)[0]):
    if y_train[i] == LuckyNumber:
        y_train[i] = 1
    else:
        y_train[i] = -1

for i in range(0, np.shape(y_test)[0]):
    if y_test[i] == LuckyNumber:
        y_test[i] = 1
    else:
        y_test[i] = -1

# visualize some images of the dataset with the new labels
if FIGURE:
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(y_train[i])
        plt.show()

# Reshape of the input data from a matrix [28 x 28] to a vector [ 784 x 1 ]
x_train_vct = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test_vct = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

if BALANCING:
    rus = RandomUnderSampler()
    x_train_vct, y_train = rus.fit_resample(x_train_vct, y_train)
    x_train_vct, _, y_train, _ = train_test_split(x_train_vct, y_train, test_size=0.01)

    print('Resampled dataset shape %s' % Counter(y_train))

    if FIGURE:
        plt.figure()
        for i in range(100):
            plt.subplot(10,10,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(np.reshape(x_train_vct[i], (28, 28)))
            plt.xlabel(y_train[i])
        plt.show()

data_point = np.zeros((NN, dim_train_agent, np.shape(x_train_vct)[1]))
label_point = np.zeros((NN, dim_train_agent))

data_test = np.zeros((NN, dim_test_agent, np.shape(x_test_vct)[1]))
label_test = np.zeros((NN, dim_test_agent))

# data_validation, label_validation
for agent in range(NN):
    agent_index = agent * dim_train_agent + np.arange(dim_train_agent)
    data_point[agent, :, :] = x_train_vct[agent_index, :]
    label_point[agent, :] = y_train[agent_index]

    agent_index = agent * dim_test_agent + np.arange(dim_test_agent)
    data_test[agent, :, :] = x_test_vct[agent_index, :]
    label_test[agent, :] = y_test[agent_index]

#  Set Up the Neural Network

d = [784, 784, 784, 784]
T = len(d)  # how much layer we have
dim_layer = d[0]  # number of neurons considering bias

## ALGORITHM ##
uu = np.zeros((NN, max_iters, T - 1, dim_layer, dim_layer + 1))  # +1 means bias
uu[:,0,:,:,:] = np.random.randn(NN, T-1, dim_layer, dim_layer +1)
yy = np.zeros((NN, max_iters, T - 1, dim_layer, dim_layer + 1))
grad_u = np.zeros((NN, max_iters, T - 1, dim_layer, dim_layer + 1))  # +1 means bias

# force the last layer to have a 1 and 0 ... 0
# uu[agent, iteration, layer, neuron, neuron + bias]
uu[:, 0, -1, 1:] = np.zeros((785))
JJ = np.zeros((NN, max_iters))
dJJ = np.zeros((NN, max_iters))


## ITERATION 0 - Initialization of Gradient of u
for agent in range(NN):
    print("Agent {}, iter = {}".format(agent, 0))
    for image in range(dim_train_agent):
        temp_data = data_point[agent, image, :]
        temp_label = label_point[agent, image]

        temp_data_test = data_test[agent, image, :]
        temp_label_test = label_test[agent, image]

        xx = forward_pass(uu[agent, 0], temp_data, T, dim_layer)
        xx_test = forward_pass(uu[agent, 0], temp_data_test, T, dim_layer)

        _, lambda_T = cost_function(xx[-1], temp_label)
        JJ_temp, _ = cost_function(xx_test[-1], temp_label_test)
        JJ[agent, 0] += (1/dim_test_agent)*JJ_temp

        delta_u = backward_pass(xx, uu[agent, 0], lambda_T, T, dim_layer)

        for layer in range(T - 1):
            grad_u[agent, 0, layer] += delta_u[layer] / (np.shape(temp_data)[0])
            yy[agent, 0, layer] += delta_u[layer] / (np.shape(temp_data)[0])

    for layer in range(T - 1):
        uu[agent, 1, layer] += WW[agent, agent] * uu[agent, 0, layer] - stepsize * grad_u[agent, 0, layer]

# ALGORITHM STARTING FROM k = 1
for iter in range(1, max_iters-1):
    for agent in range(NN):
        print("Agent {}, iter = {}".format(agent, iter))
        for image in range(dim_train_agent):
            temp_data = data_point[agent, image, :]
            temp_label = label_point[agent, image]

            temp_data_test = data_test[agent, image, :]
            temp_label_test = label_test[agent, image]

            xx = forward_pass(uu[agent, iter], temp_data, T, dim_layer)
            xx_test = forward_pass(uu[agent, iter], temp_data_test, T, dim_layer)

            _, lambda_T = cost_function(xx[-1], temp_label)

            JJ_temp, _ = cost_function(xx_test[-1], temp_label_test)
            JJ[agent, iter] += (1/dim_test_agent)*JJ_temp
            # dJJ[agent, iter] += np.linalg.norm(dJJ_temp)

            delta_u = backward_pass(xx, uu[agent, iter], lambda_T, T, dim_layer)
            for layer in range(T - 1):
                grad_u[agent, iter, layer] += delta_u[layer] / (np.shape(temp_data)[0])

    ## Gradient Tracking
    for agent in range(NN):
        print("Agent = {}, Gradient Tracking".format(agent))
        for layer in range(T - 1):
            delta_grad_u = grad_u[agent, iter, layer] - grad_u[agent, iter - 1, layer]

            yy[agent, iter, layer] = WW[agent, agent] * yy[agent, iter - 1, layer] + delta_grad_u

            for neigh in G.neighbors(agent):
                yy[agent, iter, layer] = WW[agent, neigh] * yy[neigh, iter - 1, layer]

            uu[agent, iter + 1, layer] = WW[agent, agent] * uu[agent, iter, layer] - stepsize * yy[agent, iter, layer]

            for neigh in G.neighbors(agent):
                uu[agent, iter + 1, layer] += WW[agent, neigh] * uu[neigh, iter, layer]
        dJJ[agent, iter] += (JJ[agent, iter] - JJ[agent, iter-1])/iter
if SAVE:
    np.save('store_data/max_iter.npy', max_iters, allow_pickle=True)
    np.save('store_data/NN.npy', NN, allow_pickle=True)
    np.save('store_data/JJ.npy', JJ, allow_pickle=True)
    np.save('store_data/dJJ.npy', dJJ, allow_pickle=True)
    np.save('store_data/uu.npy',uu, allow_pickle=True)
    np.save('store_data/yy.npy',yy, allow_pickle=True)


plt.figure()
plt.plot(range(max_iters-1), (JJ[0,:-1]))
plt.title("Cost function over iterations")
plt.grid()

plt.figure()
plt.plot(range(max_iters-1), dJJ[0,:-1])
plt.title("Gradient of Cost function")
plt.grid()

val_function(uu[0,-1], x_test_vct, y_test, T, dim_layer, dim_test_agent)

print('DAJE TUTTO OK')
