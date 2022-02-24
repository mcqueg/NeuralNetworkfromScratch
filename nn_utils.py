# helpful functions for nn.py

import numpy as np


# activations
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    assert(A.shape == Z.shape)

    return A, cache


def relu(Z):
    A = Z * (Z > 0)
    cache = Z

    assert(A.shape == Z.shape)

    return A, cache


def sigmoid_back(dA, cache):
    '''
    Arguments:
        dA -- post-activation gradient
        cache -- stores 'Z' value from forward activation

    Returns:
        dZ -- gradient of cost with respect to Z
    '''
    Z = cache

    x = 1/(1+np.exp(-Z))
    dZ = dA * x * (1-x)

    assert (dZ.shape == Z.shape)

    return dZ


def relu_back(dA, cache):
    '''
    Arguments:
        dA -- post-activation gradient
        cache -- stores 'Z' value from forward activation

    Returns:
        dZ -- gradient of cost with respect to Z
    '''
    Z = cache

    dZ = np.array(dA, copy=True)  # converting dz to a correct object.

    # When z <= 0, dz is set to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ
