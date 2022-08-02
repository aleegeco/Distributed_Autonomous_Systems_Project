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
FIGURE = True
PRINT = False
SAVE = True
# chosen digit to classify
LuckyNumber = 6

epochs = 200
stepsize = 0.01

# Graph parameters - To simulate distributed behaviour
NN = 4  # Number of agents
p_ER = 0.4  # Probability of spawning an edge
I_NN = np.identity(NN, dtype=int)  # necessary to build the Adj

dim_train_agent = 50  # Images for each agent
dim_test_agent = int(0.5*dim_train_agent) # test Images for each agent

# Data acquisition and processing from MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# adjusting the type of the data contained in the arrays in this way they can also be negative
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

# Normalization of the images between [0, 1] otherwise they saturate the activation function
x_train = x_train / 255
x_test = x_test / 255

#  Set Up the Neural Network
d = [784, 784, 784, 784]  # Define the number of neurons for each layer (constant
T = len(d)  # Layers of the neural network
dim_layer = d[0]  # number of neurons considering bias
print(f'Settings: NN={NN}, n_images={dim_train_agent}, epochs={epochs}, stepsize={stepsize}, layers={T} ')

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

WW = np.zeros((NN, NN))  # Initializing weighting matrix
# For cycle to compute the weighted adjacency matrix
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

# Creating the Graph G
G = nx.from_numpy_array(Adj)
if FIGURE:
    nx.draw(G, with_labels=True, font_weight='bold', node_color='orange', node_size=800)
    plt.show()

# Associate 0 to data which not represent the number we want to classify
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

    # Shuffling the train and the test set
    x_train_vct, _, y_train, _ = train_test_split(x_train_vct, y_train, test_size=0.01)
    x_test_vct, _, y_test, _ = train_test_split(x_test_vct, y_test, test_size=0.01)
    if PRINT:
        print('Resampled dataset shape %s' % Counter(y_train))

# Initializing the store to split the train set between agents
data_train = np.zeros((NN, dim_train_agent, np.shape(x_train_vct)[1]))
label_train = np.zeros((NN, dim_train_agent))

# Initializing the store to split the test set between agents
data_test = np.zeros((NN, dim_test_agent, np.shape(x_test_vct)[1]))
label_test = np.zeros((NN, dim_test_agent))

# Splitting the train and the test set between agents
for agent in range(NN):
    agent_index = agent * dim_train_agent + np.arange(dim_train_agent)
    data_train[agent, :, :] = x_train_vct[agent_index, :]
    label_train[agent, :] = y_train[agent_index]

    agent_index = agent * dim_test_agent + np.arange(dim_test_agent)
    data_test[agent, :, :] = x_test_vct[agent_index, :]
    label_test[agent, :] = y_test[agent_index]

# Initialization of the weights
uu = np.zeros((epochs, NN, T-1, dim_layer, dim_layer+1))
uu[0] = np.random.randn(NN, T-1, dim_layer, dim_layer+1)  # First iteration has random weights
uu[:, :, -1, 1:] = 0  # Resetting the weights of the last layer except the first neuron to zero

zz = np.zeros_like(uu)  # initialization of z for distributed tracking with same dimension of u
delta_u = np.zeros_like(uu) # initialization of delta_u with same dimension of u

J = np.zeros((epochs, NN))

# Useful prints
print(f'k\t', end='')
for agent in range(NN):
    if agent == NN-1:
        print(f'J[{agent}]\t \t d_u[{agent}]\t \t')
    else:
        print(f'J[{agent}]\t \t d_u[{agent}]\t \t', end='\t \t')

# START THE ALGORITHM
for k in range(epochs-1):
    for agent in range(NN):
        # Neural Network evaluation
        for image in range(dim_train_agent):
            temp_data = data_train[agent, image]  # Pick the image fo the forward pass
            temp_label = label_train[agent, image]  # Pick the right label to evaluate the cost function

            xx = forward_pass(uu[k, agent], temp_data, T, dim_layer)

            J_temp, lambdaT = cost_function(xx[-1, 0], temp_label)
            J[k, agent] += J_temp
            temp_delta_u = backward_pass(xx, uu[k, agent], lambdaT, T, dim_layer)
            delta_u[k, agent] += temp_delta_u

    # Causal Gradient Tracking Algorithm
    print(f'{k}\t', end='')
    for agent in range(NN):
        uu[k+1, agent] = WW[agent, agent]*uu[k, agent]
        for neigh in G.neighbors(agent):
            uu[k+1, agent] += WW[agent, neigh]*uu[k, neigh]
        uu[k+1, agent] += zz[k, agent] - stepsize*delta_u[k, agent]

        zz[k+1, agent] = WW[agent, agent]*zz[k, agent] - stepsize*WW[agent, agent]*delta_u[k, agent]
        for neigh in G.neighbors(agent):
            zz[k+1, agent] += WW[agent, neigh]*zz[k, neigh] - stepsize*WW[agent, neigh]*delta_u[k, neigh]
        zz[k+1, agent] += stepsize*delta_u[k, agent]  # not sure that this is needed

        if agent == NN - 1:
            print(f'{J[k, agent]:4.2f}\t \t{np.linalg.norm(delta_u[k, agent]):4.2f}\t \t')
        else:
            print(f'{J[k, agent]:4.2f}\t \t{np.linalg.norm(delta_u[k, agent]):4.2f}\t \t', end='\t \t')

# Save the results to plot them easily by using the "plot.py" executable
if SAVE:
    np.save('store_data/epochs.npy', epochs, allow_pickle=True)
    np.save('store_data/NN.npy', NN, allow_pickle=True)
    np.save('store_data/JJ.npy', J, allow_pickle=True)
    np.save('store_data/delta_u.npy', delta_u, allow_pickle=True)
    np.save('store_data/uu.npy', uu, allow_pickle=True)


# simple plot of the cost function over the epochs
_, ax = plt.subplots()
ax.plot(range(epochs), J)
ax.title.set_text('J')
plt.show()

# validation function which computes and print the % of correct classification with some other useful info
val_function(uu, data_test, label_test, dim_test_agent, NN, dim_layer, T)


print('DAJE TUTTO OK')
