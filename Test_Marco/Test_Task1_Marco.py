import tensorflow as tf
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # this library will be used for data visualization
import networkx as nx  # library for network creation/visualization/manipulation
import Function_Task_1

np.random.seed(0)

PRINT = True
FIGURE = False

# Parameters for the data size
test_set_size = 0.1  # percentage of the test set over the entire data
percent = 1  # percentage of data we want to give to our system from all the data available

# chosen digit to work on
LuckyNumber = 4

#       DEFINITION OF THE BINOMIAL GRAPH

NN = 7  # number of AGENTS
p_ER = 0.3  # spawn edge probability
I_NN = np.identity(NN, dtype=int)  # necessary to build the Adj

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

#                                       Compute mixing matrices
#                     Metropolis-Hastings method to obtain a doubly-stochastic matrix

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

if PRINT: print('Check Stochasticity\n row:    {} \n column: {}'.format(np.sum(WW, axis=1), np.sum(WW, axis=0)))

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
x_total_temp = np.append(x_train, x_test, axis=0)
x_total = x_total_temp[0: int(np.shape(x_total_temp)[0] * percent)]
y_total_temp = np.append(y_train, y_test, axis=0)
y_total = y_total_temp[0: int(np.shape(y_total_temp)[0] * percent)]

# Random redistribution of the data in two sets ( test and train)
x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=test_set_size)

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

# Reshape of the input data from a vecto to a matrix
x_train_vct = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
x_test_vct = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))






print('DAJE TUTTO OK')

# MANCA LA PARTE DI BILANCIAMENTO DEL DATASET

