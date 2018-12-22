"""
MultilayerPerceptron taken from https://github.com/alirezamika/evostra/blob/master/evostra/models/feed_forward_network.py
"""

import numpy as np

class MultilayerPerceptron(object):
    """
    Architecture:
    INPUT -> HIDDEN -> ... -> HIDDEN -> OUTPUT
    """
    def __init__(self, input_size, output_size):
        # import mlp parameters
        from config import mlp_params
        self.layer_activation = mlp_params["layer_activation"]
        self.use_bias = mlp_params["use_bias"]
        kernel_initializer = mlp_params["kernel_initializer"]
        bias_initializer = mlp_params["bias_initializer"]
        layers = [input_size] + mlp_params["hidden_layers"] + [output_size]

        # initialize weights and biases
        self.w, self.b = [], []
        for i in range(len(layers) - 1):
            self.w.append(kernel_initializer(shape=(layers[i], layers[i+1])))
            self.b.append(bias_initializer(shape=(layers[i+1])))

    def predict(self, inp):
        out = inp.flatten()
        for w, b in zip(self.w, self.b):
            out = np.matmul(w.T, out) + b
            out = self.layer_activation(out)
        return np.clip(out, -1, 1)

    def get_weights(self):
        if self.use_bias:
            return self.w + self.b
        else:
            return self.w

    def set_weights(self, weights):
        if self.use_bias:
            l = len(weights) // 2
            self.w = weights[:l]
            self.b = weights[l:]
        else:
            self.w = weights