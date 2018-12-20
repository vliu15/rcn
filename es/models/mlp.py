"""
MultilayerPerceptron taken from https://github.com/alirezamika/evostra/blob/master/evostra/models/feed_forward_network.py
"""

import numpy as np
import copy

class MultilayerPerceptron(object):
    """
    Architecture:
    INPUT -> HIDDEN -> ... -> HIDDEN -> OUTPUT
    """
    def __init__(self, input_size, output_size):
        # import mlp parameters
        from es.config import mlp_params
        self.layer_activation = mlp_params["layer_activation"]
        kernel_initializer = mlp_params["kernel_initializer"]
        layers = [input_size] + mlp_params["hidden_layers"] + [output_size]

        # initialize weights
        self.weights = []
        for i in range(len(layers) - 1):
            self.weights.append(kernel_initializer(shape=(layers[i], layers[i + 1])))

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        for layer in self.weights:
            out = np.dot(out, layer)
            out = self.layer_activation(out)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

class ParallelMultilayerPerceptron(MultilayerPerceptron):
    """
    See MultilayerPerceptron for MLP architecture.
    Architecture:
    INPUT -> LINEAR
    INPUT -> MLP
        ...
    INPUT -> MLP
    ----
    LINEAR + MLP + ... + MLP -> OUTPUT
    """
    def __init__(self, input_size, output_size):
        # import mlp parameters
        from es.config import pmlp_params
        from es.config import lin_params
        kernel_initializer = lin_params["kernel_initializer"]
        # intialize linear weights
        self.linear_weights = [kernel_initializer(shape=(input_size, 1))]
        # initialize mlp weights: list of mlp objects
        self.blocks = []
        for _ in range(pmlp_params["num_blocks"]):
            self.blocks.append(MultilayerPerceptron(input_size, output_size))
    
    def predict(self, inp):
        inp = np.expand_dims(inp.flatten(), 0)
        outputs = []
        # linear output:
        lin_out = np.dot(inp, self.linear_weights)
        outputs.append(lin_out)
        for block in self.blocks:
            mlp_out = inp
            # mlp output:
            for layer in block.weights:
                mlp_out = np.dot(mlp_out, layer)
                mlp_out = block.layer_activation(mlp_out)
            outputs.append(mlp_out)
        
        return sum(outputs)

    def get_weights(self):
        weights = []
        for block in self.blocks:
            weights.extend(block.weights)
        weights.extend(self.linear_weights)
        return weights

    def set_weights(self, weights):
        idx = 0
        for block in self.blocks:
            # extract weights of mlp
            block.weights = weights[idx:idx+len(block.weights)]
            idx += len(block.weights)
        # extract weights of linear
        self.linear_weights = weights[idx:]
        