import numpy as np

###############
# activations #
###############


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    assert(A.shape == Z.shape)
    return A


def sigmoid_derivative(Z):
    dA = np.exp(-Z) / ((1 + np.exp(-Z)) ** 2)
    assert (dA.shape == Z.shape)
    return dA


def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    return A


def relu_derivative(Z):
    dA = 1 * (Z > 0)
    assert (dA.shape == Z.shape)
    return dA


def tanh(Z):
    A = np.tanh(Z)
    assert (A.shape == Z.shape)
    return A


def tanh_derivative(Z):
    dA = 1.0 - np.tanh(Z) ** 2
    assert (dA.shape == Z.shape)
    return dA


def softmax(Z):
    log_c = np.max(Z, axis=0, keepdims=True)
    Z_exp = np.exp(Z - log_c)
    Z_sum = np.sum(Z_exp, axis=0, keepdims=True)
    A = np.divide(Z_exp, Z_sum)
    assert(A.shape == Z.shape)
    return A


def softmax_derivative(Z):
    Z_exp = np.exp(Z)
    Z_sum = np.sum(Z_exp, axis=0, keepdims=True)
    A = np.divide(Z_exp, Z_sum)
    dA = A * (1 - Z_sum)
    assert(dA.shape == Z.shape)
    return dA
