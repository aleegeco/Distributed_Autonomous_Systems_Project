import numpy as np


def sigmoid_fn(xi):
    return 1 / (1 + np.exp(-xi))


def sigmoid_fn_derivative(xi):
    return sigmoid_fn(xi) * (1 - sigmoid_fn(xi))


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

        xtp[ell] = sigmoid_fn(temp)  # x' * u_ell

    return xtp


# Forward Propagation
def forward_pass(uu, x0, T, d):
    """
        input:
              uu input trajectory: u[0],u[1],..., u[T-1]
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
        dsigma_j = sigmoid_fn_derivative(xt @ ut[j, 1:] + ut[j, 0])

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
