# helpful functions for nn.py

import numpy as np
import math


# activations
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = Z * (Z > 0)
    cache = Z
    return A, cache
