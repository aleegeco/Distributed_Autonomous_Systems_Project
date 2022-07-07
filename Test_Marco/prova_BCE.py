import numpy as np

def bce(pred, real):
    J = real*np.log(pred) + (1-real)*np.log(1-pred)
    dJ = real/pred + (1-real)/(1-pred)

    return J, dJ

xx = -np.ones((10))
real = 1

J, dJ = bce(xx, real)