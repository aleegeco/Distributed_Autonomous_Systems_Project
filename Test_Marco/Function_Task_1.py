import numpy as np
from ActivationFunctions import tanh as act_function
from ActivationFunctions import tanh_derivative as der_ac_function


# Inference: x_tp = f(xt,ut)
def inference_dynamics(xt, ut):
    """
        input:
              xt current state
              ut current input - layer
        output:
              xtp next state
    """
    dim_layer = np.shape(ut)[0]

    xtp = np.zeros(dim_layer)
    for ell in range(dim_layer):
        temp = xt @ ut[ell, 1:] + ut[ell, 0]  # including the bias

        xtp[ell] = act_function(temp)  # x' * u_ell

    return xtp


# Forward Propagation
def forward_pass(uu, x0, T, d):
    """
        input:
              uu input trajectory: u^k[0],u^k[1],..., u^k[T-1]
              x0 initial condition
              T. number of layers
              d number of neurons
        output:
              xx state trajectory: x[1],x[2],..., x[T]
    """
    xx = np.zeros((T, d))
    xx[0] = x0

    for t in range(T - 1):
        xx[t + 1] = inference_dynamics(xx[t], uu[t])  # x^+ = f(x,u)

    return xx


# Adjoint dynamics:
#   state:    lambda_t = A.T lambda_tp
#   output: deltau_t = B.T lambda_tp
def adjoint_dynamics(ltp, xt, ut, d):
    """
        input:
              llambda_tp current costate
              xt current state
              ut current input
        output:
              llambda_t next costate
              delta_ut loss gradient wrt u_t
    """
    df_dx = np.zeros((d, d))

    # df_du = np.zeros((d,(d+1)*d))
    Delta_ut = np.zeros((d, d + 1))

    for j in range(d):
        dsigma_j = der_ac_function(xt @ ut[j, 1:] + ut[j, 0])

        df_dx[:, j] = ut[j, 1:] * dsigma_j
        # df_du[j, XX] = dsigma_j*np.hstack([1,xt])

        # B'@ltp
        Delta_ut[j, 0] = ltp[j] * dsigma_j
        Delta_ut[j, 1:] = xt * ltp[j] * dsigma_j

    lt = df_dx @ ltp  # A'@ltp
    # Delta_ut = df_du@ltp

    return lt, Delta_ut


# Backward Propagation
def backward_pass(xx, uu, llambdaT, T, d):
    """
        input:
              xx state trajectory: x[1],x[2],..., x[T]
              uu input trajectory: u[0],u[1],..., u[T-1]
              llambdaT terminal condition
        output:
              llambda costate trajectory
              delta_u costate output, i.e., the loss gradient
    """

    llambda = np.zeros((T, d))
    llambda[-1] = llambdaT

    Delta_u = np.zeros((T - 1, d, d + 1))

    for t in reversed(range(T - 1)):  # T-2,T-1,...,1,0
        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t], d)

    return Delta_u


def cost_function(predicted: np.array, label: int):
    J = (predicted - label) @ (predicted - label).T
    grad_J = 2 * (predicted - label)
    return J, grad_J


def ValidationFunction(uu: np.array, VectoryzedImagesTestArray: np.array, LablesTestArray: np.array,
                       NumberOfEvaluations, fringe=0.0):
    """
        Input:
            uu : tensor containing the weights of the neural network
            VectoryzedImagesTestArray : vector containing the vectorized image
            LablesTestArray : vector containing the lables of the images
            NumberOfEvaluations : Number of images we want to lable
            fringe : fringe above which we consider the prediction

        Output:
            Resoult : Dictionary containing the number of correct/wrong predictions and False positive/negative

        Dictionary values
            1 -> the NN predict 1 as lable and the true lable is 1 ( correct lable for 1)
           -1 -> the NN predict -1 as lable and the true lable is -1 ( correct lable for -1)
            2 -> the NN predict 1 as lable and the true lable is -1 ( false positive )
           -2 -> the NN predict -1 as lable and the true lable is 1 ( false negative )
            0 -> the prediction is under the treshold

    """
    VectorOfEstimation = -np.ones(NumberOfEvaluations)
    for i in range(NumberOfEvaluations):
        xx = forward_pass(uu, VectoryzedImagesTestArray[i])  # here i move forward the image in the neural network
        # prediction = np.mean(xx[-1][:])# here i compute the avarege of the resoults of neural network
        prediction = xx[-1][0]

        if (prediction >= fringe) and (LablesTestArray[i] == 1):  # prediction = 1 , true_lable = 1
            VectorOfEstimation[i] = 1

        elif (prediction <= -fringe) and (LablesTestArray[i] == -1):  # prediction = -1 , true_lable = -1
            VectorOfEstimation[i] = -1

        elif ((prediction >= fringe) and (LablesTestArray[i] == -1)):  # prediction = 1 , true_lable = -1
            VectorOfEstimation[i] = 2

        elif ((prediction <= -fringe) and (LablesTestArray[i] == 1)):
            VectorOfEstimation[i] = -2

        else:
            VectorOfEstimation[i] = 0

    unique, counts = np.unique(VectorOfEstimation, return_counts=True)
    Result = dict(zip(unique, counts))
    return Result

def Results(Dictionary, samples):
    '''
        Input:
            Dictionary: contains the number of guesses associated with each category
            samples: number of samples (images) used for the evaluation

        Categories:(intended as dictionary keys)
            1 -> the NN predict 1 as lable and the true lable is 1 ( correct lable for 1)
           -1 -> the NN predict -1 as lable and the true lable is -1 ( correct lable for -1)
            2 -> the NN predict 1 as lable and the true lable is -1 ( false positive )
           -2 -> the NN predict -1 as lable and the true lable is 1 ( false negative )
            0 -> the prediction is under the treshold
    '''
    if "1.0" in Dictionary.keys():  # to key "1.0" is associated the number of elements of category 1
        Category1 = Dictionary["1.0"]
    else:
        Category1 = 0

    if "-1.0" in Dictionary.keys():  # to key "-1.0" is associated the number of elements of category -1
        Category2 = Dictionary["-1.0"]
    else:
        Category2 = 0

    if "2.0" in Dictionary.keys():  # to key "2.0" is associated the number of elements of category 2
        Category3 = Dictionary["2.0"]
    else:
        Category3 = 0

    if "-2.0" in Dictionary.keys():  # to key "-2.0" is associated the number of elements of category -2
        Category4 = Dictionary["-2.0"]
    else:
        Category4 = 0

    if "0.0" in Dictionary.keys():  # to key "0.0" is associated the number of elements of category 0
        Category5 = Dictionary["-2.0"]
    else:
        Category5 = 0

    print("The accuracy is {} % where:\n".format((Category1 + Category2) / samples * 100))  # sum of first and second category expressed in percentage
    print("\tFalse positives {} \n".format(Category3))  # third category ( false positive)
    print("\tFalse negatives {} \n".format(Category4))  # fourth category ( false negative)
    print("\tNumber of times LukyNumber has been identified correctly {}\n".format(Category1))  # first category ( images associated to lable 1 predicted correctly )
    print("\tNumber of times non LukyNumber has been identified correctly {}\n".format(Category2))  # second category ( images associated to lable -1 predicted correctly