import numpy as np
from models.mlp import MultilayerPerceptron
from models.scn import StructuredControlNet
from models.rnn import RecurrentNeuralNetwork
from models.rcn import RecurrentControlNet
from models.tdnn import TimeDelayNeuralNetwork
from models.tdcn import TimeDelayControlNet
import utils.activations as a
import utils.initializers as i

# select model from command line arg
map_str_model = {
    # baselines
    'mlp': MultilayerPerceptron,
    'scn': StructuredControlNet,

    # custom models
    'rnn': RecurrentNeuralNetwork,
    'rcn': RecurrentControlNet,
    'tdnn': TimeDelayNeuralNetwork,
    'tdcn': TimeDelayControlNet
}

## ==== BASELINE MODELS ==== ##
# openai baseline mlp-64
mlp_params = {
    'layer_activation': np.tanh,
    'hidden_layers': [64, 64],
    'kernel_initializer': i.constant(0),
    'bias_initializer': i.constant(0),
    'use_bias': False
}
# structured control net baseline
scn_params = {
    # nonlinear module
    'layer_activation': np.tanh,
    'hidden_layers': [16, 16],
    'n_kernel_initializer': i.constant(0),
    'n_bias_initializer': i.constant(0),
    'n_use_bias': False,

    # linear module
    'l_kernel_initializer': i.constant(0),
    'l_bias_initializer': i.constant(0),
    'l_use_bias': False
}

## ==== RECURRENT MODELS ==== ##
# base recurrent neural network
rnn_params = {
    'layer_activation': np.tanh,
    'hidden_size': 32,
    'kernel_initializer': i.constant(0),
    'bias_initializer': i.constant(0),
    'use_bias': True
}
# recurrent control net
rcn_params = {
    # nonlinear module
    'layer_activation': np.tanh,
    'hidden_size': 32,
    'n_kernel_initializer': i.constant(0),
    'n_bias_initializer': i.constant(0),
    'n_use_bias': True,

    # linear module
    'l_kernel_initializer': i.constant(0),
    'l_bias_initializer': i.constant(0),
    'l_use_bias': True
}

## ==== TIME DELAY MODELS ==== ##
# time delay neural network
tdnn_params = {
    'layer_activation': np.tanh,
    'stride': 1,
    'window': 15,
    'layers': [32],
    'kernel_initializer': i.constant(0),
    'bias_initializer': i.constant(0),
    'use_bias': True
}
# time delay control net
tdcn_params = {
    # nonlinear module
    'layer_activation': np.tanh,
    'stride': 1,
    'window': 15,
    'layers': [16],
    'n_kernel_initializer': i.constant(0),
    'n_bias_initializer': i.constant(0),
    'n_use_bias': True,

    # linear module
    'l_kernel_initializer': i.constant(0),
    'l_bias_initializer': i.constant(0),
    'l_use_bias': True
}
