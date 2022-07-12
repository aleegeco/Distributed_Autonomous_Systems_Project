import numpy as np
from ActivationFunctions import sigmoid_fn as act_function
from ActivationFunctions import sigmoid_fn_derivative as der_ac_function


# Inference: x_tp = f(xt,ut)
def inference_dynamics(xt, ut):
    """
        input:
              xt current state
              ut current input - layer
        output:
              xtp next state
    """

    dim_layer = np.shape(xt)[0]
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
        if t == T-2:
            temp = xx[t] @ uu[t,0, 1:] + uu[t,0, 0]
            xx[t + 1, 0] = act_function(temp)
        else:
            xx[t + 1] = inference_dynamics(xx[t], uu[t])  # x^+ = f(x,u)

    return xx


# Adjoint dynamics:
#   state:    lambda_t = A.T lambda_tp
#   output: deltau_t = B.T lambda_tp
def adjoint_dynamics(ltp, xt, ut, d, t, T):
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
    df_du = np.zeros((d, (d+1)))

    Delta_ut = np.zeros((d, d + 1))

    if t == T-2:
        dsigma_j = der_ac_function(xt @ ut[0, 1:] + ut[0, 0])

        df_dx[:, 0] = ut[0, 1:] * dsigma_j
        # df_du[j, XX] = dsigma_j*np.hstack([1,xt])

        # B'@ltp
        Delta_ut[0, 0] = ltp[0] * dsigma_j
        Delta_ut[0, 1:] = xt * ltp[0] * dsigma_j

    else:
        for j in range(d):
            dsigma_j = der_ac_function(xt @ ut[j, 1:] + ut[j, 0])

            df_dx[:, j] = ut[j, 1:] * dsigma_j
            df_du[j, :] = np.hstack([1, xt])*dsigma_j

            # B'@ltp
            Delta_ut[j, 0] = df_du[j, 0]*ltp[j]
            Delta_ut[j, 1:] = df_du[j, 1:]*ltp[j]

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

    for t in reversed(range(T - 1)):  # T-2,...,1,0

        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t], d, t, T)

    return Delta_u


def MSE(predicted: int, label: int):
    J = (predicted - label)**2
    grad_J = 2 * (predicted - label)
    return J, grad_J

def BCE(predicted: int, label: int):
    J = - (label * np.log(predicted + 1e-10) + (1 - label) * np.log(1 - predicted + 1e-10))
    grad_J = - label / (predicted + 1e-10) + (1 - label) / (1 - predicted + 1e-10)
    return J, grad_J


def val_function(uu, x_test_vct, y_test, T, dim_layer, dim_test):
    counter_corr_label = 0
    correct_predict = 0
    correct_predict_not_lucky = 0
    false_positive = 0
    false_negative = 0
    for image in range(dim_test):
        xx = forward_pass(uu, x_test_vct[image, :], T, dim_layer)
        predict = xx[-1,0]
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

    print("The accuracy is {} % where:\n".format((correct_predict + correct_predict_not_lucky) / dim_test * 100))  # sum of first and second category expressed in percentage
    print("\tFalse positives {} \n".format(false_positive))  # third category ( false positive)
    print("\tFalse negatives {} \n".format(false_negative))  # fourth category ( false negative)
    print("\tNumber of times LuckyNumber has been identified correctly {} over {} \n".format(correct_predict, dim_test))  # first category ( images associated to lable 1 predicted correctly )
    print("\tNumber of times not LuckyNumber has been identified correctly {} over {} \n".format(correct_predict_not_lucky, dim_test))  # first category ( images associated to lable 1 predicted correctly )
    print("The effective LuckyNumbers in the tests are: {}".format(counter_corr_label))
    return None
