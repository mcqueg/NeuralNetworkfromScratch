
# implenmentation of a neural network using pure numpy


import numpy as np
from nn_utils import *

# Initialize parameters
def initialize_params(layer_dims):
    '''
    Arguments:
        layer_dims -- array containing dimensions for each layer of the network

    Returns:
        params -- dictionary containing the parameters that have been randomly
        initialized.
            Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
            bl -- bias matrix with shape (layer_dims[l], 1)
    '''

    np.random.seed(3)
    params = {}
    L = len(layer_dims)  # gets the number of layers in the network

    for l in range(1, L):  # start at 1 to avoide index/layer num confusion

        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

        # check to makes ure the shapes match
        assert(params['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(params['b' + str(l)].shape == (layer_dims[l], 1))

        return params


# Forward prop
# linear_forward step for one layer

def linear_forward(A, W, b):
    '''
    Arguments:
        A -- activation matrix from the previous layer or input.
             size (Al-1, num examples)
        W -- weights matrix
            size (Wl, Wl-1)
        b -- bias vector
            size (bl, 1)

    Returns:
        Z -- this will be the input to the activation layer
        cache -- tuple containing A, W, b..used in back prop
    '''
    # matrix multiplication
    Z = np.dot(W, A) + b
    # cache of forward prop parameters to use in back prop
    cache = (A, W, b)


# activation_forward step for one layer(uses linear_forward)
def linear_activation_forward(A_prev, W, b, activation):

    '''
    Arguments:
        A -- activation matrix from the previous layer or input.
             size (Al-1, num examples)
        W -- weights matrix
            size (Wl, Wl-1)
        b -- bias vector
            size (bl, 1)
        activation -- string that specifies which activation method to use
            options include: "relu" or "sigmoid"
    Returns:
    A -- output of the activation funciton
    cache -- tuple with both "linear_cache" & "activation_cache"
    '''

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    if activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    return A, cache

# forward_prop for all layers (uses activation forward)


# cost function
# take activation from last layer and true label

# Back prop
# linear_backward step for one layer
# activation_backward for one layer (uses linear_backward)
# backward_prop for all layers (using activation backward)
# needs cache from the forward pass

# update params using gradient descent
