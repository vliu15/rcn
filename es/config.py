import numpy as np
import es.models.mlp as m
import es.models.scn as s
import es.models.frs as f
import es.models.rnn as r
import es.utils.activations as a
import es.utils.initializers as i

# select model from command line arg
map_str_model = {
    'mlp': m.MultilayerPerceptron,
    'pmlp': m.ParallelMultilayerPerceptron,

    'scn': s.StructuredControlNet,
    'dscn': s.DeepStructuredControlNet,

    'ln': f.LocomotorNet,

    'rnn': r.RecurrentNeuralNetwork,
    'gru': r.GatedRecurrentUnit,
    'lstm': r.LongShortTermMemory
}

# mlp, pmlp, scn, dscn
mlp_params = {
    'hidden_layers': [16, 16],
    'kernel_initializer': np.zeros,
    'layer_activation': np.tanh
}

# pmlp
pmlp_params = {
    'num_blocks': 4
}

# scn, dscn
lin_params = {
    'kernel_initializer': np.zeros
}

# dscn
dscn_params = {
    'num_blocks': 4,
    'use_residual': True
}

# ln
frs_params = {
    'frs_size': 16,
    'frs_func': np.sin,
    'amp_initializer': np.zeros,
    'freq_initializer': np.zeros,
    'phase_initializer': np.zeros
}

# rnn, gru, lstm
rnn_params = {
    'layer_activation': np.tanh,
    'hidden_size': 64,
    'kernel_initializer': np.zeros,
    'bias_initializer': np.zeros,
    'use_bias': True
}

# gru, lstm
gru_params = {
    'gate_activation': a.sigmoid
}

# lstm
lstm_params = {
    'cell_activation': np.tanh      # or a.linear
}
