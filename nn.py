
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


# Forwar linear step for one layer
def forward_linear(A, W, b):
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


# forward activation step for one layer(uses linear_forward)
def forward_activation(A_prev, W, b, activation):

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
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    if activation == 'relu':
        Z, linear_cache = forward_linear(A_prev, W, b)
        A, activation_cache = relu(Z)

    return A, cache


# forward_prop for all layers (uses forward_activation)
def forward_prop(X, params):
    '''
    Forward propagation through all layers. using activation_forward()
     -- Linear -> Relu x L-1 with Linear -> Sigmoid for the last layer

     Arguments:
        X -- data as a numpy array, sahpe (input size, num examples)
        parameters -- output from initialize_params()

     Returns:
        AL -- activation output from the last layer (sigmoid)
        caches -- list of caches from the forward propogation
            list of every cache from forward activation (total num = L)
    '''
    caches = []
    A = X
    L = len(params) // 2    # gets the number of layers based on the params

    # loop through layers and calculate activation for each
    # loop starts at 1 b/c 0 is the input layer
    for l in range(1, L):
        A_prev = A
        A, cache = forward_activation(A_prev, params['W' + str(l)],
                                      params['b' + str(l)], "relu")
        caches.append(cache)

    # last layer of network (L).. uses sigmoid activation
    AL, cache = forward_activation(A, params['W' + str(L)],
                                   params['b' + str(L)], "sigmoid")

    caches.append(cache)

    return AL, cache


#################
# cost function #
#################

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

# backward_linear step for one layer
def backward_linear(dZ, cache):
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


# activation in backprop for one layer (uses backward_linear)
def backward_activation(dA, cache, activation):
    '''
    Arguments:
        dA -- gradient for layer l (current layer)
        cache -- tuple storing values for linear_cache and activation_cache
        activation -- string specifying which activation is to be used

    Returns:
        dA_prev -- Gradient of cost w/ respect to activation of previous layer
        dW -- Gradient of the cost w/ respect to W
        db -- Gradient of the cost w/ respect to b
    '''
    # seperates cache from forward prop
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_back(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)

    elif activation == "relu":
        dZ = relu_back(dA, activation_cache)
        dA_prev, dW, db = backward_linear(dZ, linear_cache)

    return dA_prev, dW, db


# needs caches from the forward pass
def backward_prop(AL, Y, caches):
    '''
    Computes the backwards propagation through L-layers using its helper
    functions backward_linear() & backward_activation()

    Arguments:
        AL -- propability vector from the forward propagation
        Y -- true label vector
        caches -- list of caches
            all relu caches are in range(l,L -1)
            sigmoid cache is L - 1

    Returns:
        grads -- dictionary with gradients for each param
            dA = grads["dA" + str(l)]
            dW = grads["dW" + str(l)]
            db = grads["db" + str(l)]
    '''
    grads = {}
    L = len(caches)  # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # make the AL and Y the same shape

    # backprop initialization, derivative of cost w/ respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L-1]  # cache for the sigmoid, last layer
    dA_prev_temp, dW_temp, db_temp = backward_activation(dAL, current_cache,
                                                         "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    # loop backward through L-2 to l=0, relu layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_activation(grads["dA" + str(l+1)],
                                                             current_cache,
                                                             "relu")
        # update gradient dictionary
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


# update params using gradient descent
def update_params(params, grads, learning_rate):
    '''
    Update parameters using gradient descent

    Arguments:
        params -- python dictionary containing the parameters
        grads -- python dictionary containing grads output of backward_prop
        learning_rate -- hard coded value to be applied when updating params

    Returns:
        params -- python dictionary containing the updated parameters
    '''

    parameters = params.copy()  # copy current params for updating
    L = len(parameters) // 2  # get the number of layers in the network

    # loop through dict &  update using gradient descent
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return params


#########
# Model #
#########

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost):
    '''
    Implements a deep neural network model of size L-layers

    Arguments:
        X -- array of input data
            shape:(num_px * num_px * 3, number of examples)
        Y -- true label vector, true classification of input data
            size : numnber of examples
        layer_dims -- list containing input size & layer sizes
                w/ length of (layers + 1)
        learning_rate -- alpha value update_params during gradient descent
        num_iterations -- num iterations for optimization loop
        print_cost -- Boolean, if true prints the cost every 100 steps

    Returns
        params -- dictionary of trained weights that can be used to predict
        costs -- list containing costs from model training
    '''
    np.random.seed(1)   # for replication purposes
    costs = []  # keep track of costs for every iteration

    # Initialze the parameters
    params = initialize_params(layers_dims)

    # Gradient descent: Loop through layers for num_interations
    for i in range(0, num_iterations):

        # forward pass through netwwork, predict
        AL, caches = forward_prop(X, params)

        # compute cost from out of forward pass and true labels
        cost = compute_cost(AL, Y)

        # backward pass through netwwork, compute gradients
        grads = backward_prop(AL, Y, caches)

        # update params for gradient descent
        params = update_params(params, grads, learning_rate)

        # print cost while model runs every 100 passses and the last pass
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost afer iteration {}: {}". format(i, np.squeeze(cost)))

        # save the cost every 100 passes and the last pass
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return params, costs
