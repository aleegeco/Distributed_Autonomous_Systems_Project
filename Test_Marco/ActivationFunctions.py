import numpy as np



def sigmoid_fn(xi):
    return 1 / (1 + np.exp(-xi))


def sigmoid_fn_derivative(xi):
    return sigmoid_fn(xi + 1e-10) * (1 - sigmoid_fn(xi + 1e-10))


def ReluPlus(xi):
    if xi >= 0:
        return xi
    else:
        return 0


def ReluPlus_derivative(xi):
    if xi >= 0:
        return 1
    else:
        return 0


def sigmoid_fnPosNeg(xi):
    return 2 / (1 + np.exp(-xi + 1e-10)) - 1


# Derivative of Activation Function
def sigmoid_fn_derivativePosNeg(xi):
    return 2 * sigmoid_fn(xi) * (1 - sigmoid_fn(xi))


def tanh(xi):
    return (np.exp(xi) - np.exp(-xi)) / (np.exp(xi) + np.exp(-xi))


def tanh_derivative(xi):
    return (1 - tanh(xi)**2)
