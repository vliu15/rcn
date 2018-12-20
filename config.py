import numpy as np
import rnn.base as r
import utils.activations as a
import utils.initializers as i

# select model from command line arg
map_str_model = {
    'base': r.RecurrentNeuralNetwork
}

# base
base_params = {
    'layer_activation': np.tanh,
    'hidden_size': 32,
    'kernel_initializer': np.zeros,
    'bias_initializer': np.zeros,
    'use_bias': True
}
