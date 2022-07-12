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
FIGURE =True

# chosen digit to wor
LuckyNumber = 6

epochs = 40
stepsize = 0.01

# Data acquisition and processing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# adjusting the type of the data contained in the arrays in this way they can be also negative
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

# scale the brightness of each pixel because otherwise saturates the activation function
x_train = x_train / 255
x_test = x_test / 255

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


for k in range(epochs):
    for image in range(len(x_test_vct)):
        temp_data = x_test_vct[image]
        temp_label = y_train[image]

        lambdaT = cost_function(xx)
        xx = forward_pass(uu[k])
