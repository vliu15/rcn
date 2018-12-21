"""
File for custom layer activations.
"""

import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def linear(x):
    return x

def relu(x):
    x[x < 0] = 0
    return x
