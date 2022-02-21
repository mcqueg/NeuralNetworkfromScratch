
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


################
# Forward prop #
################


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
def forward_prop(X, params):
    '''
    Forward propagation through all layers. using linear_activation_forward()
     -- Linear -> Relu x L-1 with Linear -> Sigmoid for the last layer

     Arguments:
        X -- data as a numpy array, sahpe (input size, num examples)
        parameters -- output from initialize_params()

     Returns:
        AL -- activation output from the last layer (sigmoid)
        caches -- list of caches from the forward propogation
            list of every cache from linear_activation_forward (total num = L)
    '''
    caches = []
    A = X
    L = len(params) // 2    # gets the number of layers based on the params

    # loop through layers and calculate activation for each
    # loop starts at 1 b/c 0 is the input layer
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, params['W' + str(l)],
                                             params['b' + str(l)], "relu")
        caches.append(cache)

    # last layer of network (L).. uses sigmoid activation
    AL, cache = linear_activation_forward(A, params['W' + str(L)],
                                          params['b' + str(L)], "sigmoid")

    caches.append(cache)

    return AL, cache


# cost function
# take activation from last layer and true label
def compute_cost(AL, Y):
    '''
    computes the categorical cross entropy cost

    Arguments:
        AL -- activation from the last layer of the network (output of sigmoid)
            size: (1, number of examples)
        Y -- vector containing true labels
            size: (1, number of examples)

        Reurns:
        cost -- cross entropy cost
    '''

    m = Y.shape[1]
    cost = (-1/m) * np.sum(Y*np.log(AL) + (1-Y) * np.log(1-AL))
    cost = np.squeeze(cost)

    return cost


#############
# Back prop #
#############

# linear_backward step for one layer
def linear_backward(dZ, cache):
    '''
    Arguments:
        dZ -- gradient of cost w/ respect to linear output (current layer (l))
        cache -- tuple of (A_prev, W, b),from forward prop of current layer (l)
    Returns:
        dA_prev -- Gradient of cost w/ respect to activation of previous layer
        dW -- Gradient of the cost w/ respect to W
        db -- Gradient of the cost w/ respect to b
    '''

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    # double check that the shapes match
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

# activation_backward for one layer (uses linear_backward)
# backward_prop for all layers (using activation backward)
# needs cache from the forward pass

# update params using gradient descent
