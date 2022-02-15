
# implenmentation of a neural network using pure numpy


import numpy as np


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


dims = np.array([3, 5, 6, 1])
params = initialize_params(dims)
print(params)



# Activation Functions
# define the activation functions that could be used in forward prop
# define the derivatives of the functions for back prop

# Forward Prop
# linear_forward step for one layer
# activation_forward step for one layer(uses linear_forward)
# forward_prop for all layers (uses activation forward)
# stores values from each layer in a cache for backprop

# cost function
# take activation from last layer and true label

# Back prop
# linear_backward step for one layer
# activation_backward for one layer (uses linear_backward)
# backward_prop for all layers (using activation backward)
# needs cache from the forward pass

# update params using gradient descent
