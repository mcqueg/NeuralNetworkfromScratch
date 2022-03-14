
import numpy as np

###############
# activations #
###############


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    assert(A.shape == Z.shape)
    return A, cache


def sigmoid_derivative(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    return dZ


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    assert(A.shape == Z.shape)
    return A, cache


def relu_derivative(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def softmax(Z):
    log_c = np.max(Z, axis=0, keepdims=True)
    Z_exp = np.exp(Z - log_c)
    Z_sum = np.sum(Z_exp, axis=0, keepdims=True)
    A = np.divide(Z_exp, Z_sum)
    cache = Z
    assert(A.shape == Z.shape)
    return A, cache


'''
def tanh(Z):
    A = np.tanh(Z)
    assert (A.shape == Z.shape)
    return A


def tanh_derivative(Z):
    dA = 1.0 - np.tanh(Z) ** 2
    assert (dA.shape == Z.shape)
    return dA
'''
##################
# Model accuracy #
##################


def model_accuracy(y_hat, y_true):
    pred_labels = np.argmax(y_hat, axis=0, keepdims=True)
    frac_correct = np.mean(pred_labels == y_true) * 100
    return frac_correct
'''
################
# initializers #
################

# pass array containing the size of each layer into each initializer

# zero initialization of all params
def zero_params(layer_dims):
    parameters = {}
    L = len(layers_dims)            # number of layers in the network
    for l in range(1, L):
        parameters['W%s' % l] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b%s' % l] = np.zeros((layers_dims[l], 1))

    return parameters


# random initialization of the weight params w/ normal distribution
def random_params(layer_dims):
    parameters = {}
    L = len(layers_dims)            # number of layers in the network
    for l in range(1, L):
        # initialize weights randomly with a normal distribution
        parameters['W%s' % l] = np.random.randn((layers_dims[l], layers_dims[l-1])) * 0.1
        # initialize biases as 0
        parameters['b%s' % l] = np.zeros((layers_dims[l], 1))

        return parameters


# he initialization of the weight params
def he_params(layer_dims):
    parameters = {}
    L = len(layers_dims)            # number of layers in the network
    for l in range(1, L):
        # initialize weights randomly w/ normal distribution apply he weighting
        parameters['W%s' % l] = np.random.randn((layers_dims[l], layers_dims[l-1])) * np.sqrt(2./layers_dims[l-1])
        # initialize biases as 0
        parameters['b%s' % l] = np.zeros((layers_dims[l], 1))

        return parameters
'''