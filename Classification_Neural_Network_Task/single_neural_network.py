import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # this library will be used for data visualization
import networkx as nx  # library for network creation/visualization/manipulation
from Function_Task_1 import *
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from Function_Task_1 import MSE as cost_function

np.random.seed(0)  # generate random number (always the same seed)

BALANCING = True
FIGURE = False

# chosen digit to wor
LuckyNumber = 6

epochs = 50
stepsize = 0.01

train_images = 200
test_images = 200

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

uu = np.random.randn(epochs, T - 1, dim_layer, dim_layer + 1)
uu[:, -1, 1:] = 0
delta_u_store = np.zeros(epochs)
Delta_u = 0
J = np.zeros(epochs)

for k in range(epochs - 1):

    Delta_u = 0
    for image in range(train_images):
        temp_data = x_train_vct[image]
        temp_label = y_train[image]

        xx = forward_pass(uu[k], temp_data, T, dim_layer)

        J_temp, lambdaT = cost_function(xx[-1, 0], temp_label)
        J[k] += J_temp
        Delta_u += backward_pass(xx, uu[k], lambdaT, T, dim_layer)

    delta_u_store[k] = np.linalg.norm(Delta_u)
    uu[k + 1] = uu[k] - stepsize * Delta_u
    print(f'Iteration: {k}, Loss function: {J[k]:4.3f}, Delta_u: {delta_u_store[k]:4.3f}')

plt.semilogy(range(epochs-1), J[:-1])
plt.title('$J$')
plt.xlabel("iterations")
plt.ylabel("$J(.)$")
plt.show()

plt.plot(range(epochs), delta_u_store)
plt.title('$\Delta u$')
plt.xlabel("iterations")
plt.ylabel("$\Delta u$")
plt.show()

data_test = x_test_vct[:test_images]
label_test = y_test[:test_images]

counter_corr_label = 0
correct_predict = 0
correct_predict_not_lucky = 0
false_positive = 0
false_negative = 0
for image in range(test_images):
    xx = forward_pass(uu[-1], x_test_vct[image], T, dim_layer)
    predict = xx[-1, 0]
    if y_test[image] == 1:
        counter_corr_label += 1
    if (predict >= 0.5) and (y_test[image] == 1):
        correct_predict += 1
    elif (predict < 0.5) and (y_test[image] == 0):
        correct_predict_not_lucky += 1
    elif (predict < 0.5) and (y_test[image] == 1):
        false_negative += 1
    elif (predict >= 0.5) and (y_test[image] == 0):
        false_positive += 1


print("The accuracy is {} % where:\n".format((
                                                     correct_predict + correct_predict_not_lucky) / test_images * 100))  # sum of first and second category expressed in percentage
print("\tFalse positives {} \n".format(false_positive))  # third category ( false positive)
print("\tFalse negatives {} \n".format(false_negative))  # fourth category ( false negative)
print("\tNumber of times LuckyNumber has been identified correctly {} over {} \n".format(correct_predict,
                                                                                         counter_corr_label))  # first category ( images associated to lable 1 predicted correctly )
print("\tNumber of times not LuckyNumber has been identified correctly {} over {} \n".format(
    correct_predict_not_lucky,
    test_images - counter_corr_label))  # first category ( images associated to lable 1 predicted correctly )
print("The effective LuckyNumbers in the tests are: {}".format(counter_corr_label))