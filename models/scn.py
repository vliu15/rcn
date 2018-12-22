"""
StructuredControlNet modeled after MultilayerPerceptron.
Original implementation detailed in this blog: https://medium.com/@mariosrouji/structured-control-nets-for-deep-reinforcement-learning-tutorial-icml-published-long-talk-paper-2ff99a73c8b
"""

import numpy as np

class StructuredControlNet(object):
    """
    Nonlinear module:
        INPUT -> MLP -> OUTPUT_n

    Linear module:
        INPUT -> LIN -> OUTPUT_l
    
    Architecture:
        Nonlinear + Linear -> OUTPUT
    """

    def __init__(self, input_size, output_size):
        # import scn parameters:
        from config import scn_params
        self.layer_activation = scn_params["layer_activation"]
        self.n_use_bias = scn_params["n_use_bias"]
        self.l_use_bias = scn_params["l_use_bias"]
        n_kernel_initializer = scn_params["n_kernel_initializer"]
        l_kernel_initializer = scn_params["l_kernel_initializer"]
        n_bias_initializer = scn_params["n_bias_initializer"]
        l_bias_initializer = scn_params["l_bias_initializer"]
        layers = [input_size] + scn_params["hidden_layers"] + [output_size]

        ## ==== NONLINEAR MODULE ==== ##
        self.w_n, self.b_n = [], []
        for i in range(len(layers) - 1):
            self.w_n.append(n_kernel_initializer(shape=(layers[i], layers[i+1])))
            self.b_n.append(n_bias_initializer(shape=(layers[i+1])))

        ## ==== LINEAR MODULE ==== ##
        self.w_l = l_kernel_initializer(shape=(input_size, output_size))
        self.b_l = l_bias_initializer(shape=(output_size))

    def predict(self, inp):
        out = inp.flatten()

        ## ==== NONLINEAR MODULE ==== ##
        n_out = out
        for w, b in zip(self.w_n, self.b_n):
            n_out = np.matmul(w.T, n_out)
            n_out = self.layer_activation(n_out)

        ## ==== LINEAR MODULE ==== ##
        l_out = np.matmul(self.w_l.T, out) + self.b_l

        return np.clip(n_out + l_out, -1, 1)

    def get_weights(self):
        weights = self.w_n + [self.w_l]
        if self.n_use_bias:
            weights = weights + self.b_n
        if self.l_use_bias:
            weights = weights + [self.b_l]
        return weights

    def set_weights(self, weights):
        self.w_n = weights[:len(self.w_n)]
        self.w_l = weights[len(self.w_n)]
        if self.n_use_bias:
            self.b_n = weights[len(self.w_n)+1:len(self.w_n)+1+len(self.b_n)]
        if self.l_use_bias:
            self.b_n = weights[-1]
