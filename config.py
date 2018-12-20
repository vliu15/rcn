import numpy as np
import rnn.base as r
import rnn.rcn as s
import utils.activations as a
import utils.initializers as i

# select model from command line arg
map_str_model = {
    'rnn': r.RecurrentNeuralNetwork,
    'rcn': s.RecurrentControlNet
}

# base recurrent neural network
base_params = {
    'layer_activation': np.tanh,
    'hidden_size': 32,
    'kernel_initializer': np.zeros,
    'bias_initializer': np.zeros,
    'use_bias': True
}

# recurrent control net
rcn_params = {
    # nonlinear module
    'layer_activation': np.tanh,
    'hidden_size': 32,
    'n_kernel_initializer': np.zeros,
    'n_bias_initializer': np.zeros,
    'n_use_bias': True,

    # linear module
    'l_kernel_initializer': np.zeros,
    'l_bias_initializer': np.zeros,
    'l_use_bias': True
}