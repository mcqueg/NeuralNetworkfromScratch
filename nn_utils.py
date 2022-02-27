# helpful functions for nn.py

import numpy as np


# activations
def sigmoid(Z):
    '''
    Arguments:
        Z -- output from forward linear, dot prod of W and b:

    Returns:
        A -- post activation vector, input to next layer forward linear
        cache -- stores Z for current l in sigmoid_back
    '''
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    assert(A.shape == Z.shape)

    return A, cache


def sigmoid_back(dA, cache):
    '''
    Arguments:
        dA -- post-activation gradient
        cache -- Z value from forward activation

    Returns:
        dZ -- gradient of cost with respect to Z
    '''
    Z = cache

    x = 1/(1+np.exp(-Z))
    dZ = dA * x * (1-x)

    assert (dZ.shape == Z.shape)

    return dZ


def relu(Z):
    '''
    Arguments:
        Z -- output from forward linear, dot prod of W and b:

    Returns:
        A -- post activation vector, input to next layer forward linear
        cache -- stores Z for current l in relu_back
    '''
    A = np.maximum(0, Z)
    cache = Z

    assert(A.shape == Z.shape)

    return A, cache


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


def leaky_relu(Z):
    '''
    Arguments:
        Z -- output from forward linear, dot prod of W and b:

    Returns:
        A -- post activation vector, input to next layer forward linear
        cache -- stores Z for current l in leaky_relu_back
    '''

    A = np.maximum(0.01 * Z, Z)
    cache = Z

    assert (A.shape == Z.shape)

    return A, cache


def leaky_relu_back(dA, cache):
    '''
    Arguments:
        dA -- post-activation gradient
        cache -- stores 'Z' value from forward activation

    Returns:
        dZ -- gradient of cost with respect to Z
    '''

    Z = cache

    dZ = np.array(dA, copy=True)  # converting dz to a correct object.

    # When z <= 0 dz = 0.01
    dZ[Z <= 0] = 0.01

    return dZ


def tanh(Z):
    '''
    Arguments:
        Z -- output from forward linear, dot prod of W and b:

    Returns:
        A -- post activation vector, input to next layer forward_linear
        cache -- stores Z for current l in tanh_back
    '''
    A = tanh(Z)
    cache = Z

    assert (A.shape == Z.shape)

    return A, cache


def tanh_back(dA, cache):
    '''
    Arguments:
        dA -- post-activation gradient
        cache -- stores 'Z' value from forward_activation

    Returns:
        dZ -- gradient of cost with respect to Z
    '''
    Z = cache

    dZ = dA * (1 - np.tanh(Z)**2)

    assert (dZ.shape == Z.shape)

    return dZ
