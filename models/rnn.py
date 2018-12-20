import numpy as np
import copy

class RecurrentNeuralNetwork(object):
    """
    Architecture:
                   OUTPUT(t)
                      |
    HIDDEN(t-1) -> HIDDEN(t) -> ...
                      |
                   INPUT(t)
    """
    def __init__(self, input_size, output_size, every_t=True):
        self.every_t = every_t
        from es.config import rnn_params
        # import rnn parameters
        self.layer_activation = rnn_params["layer_activation"]
        self.use_bias = rnn_params["use_bias"]
        hidden_size = rnn_params["hidden_size"]
        kernel_initializer = rnn_params["kernel_initializer"]
        bias_initializer = rnn_params["bias_initializer"]
        # initialize rnn weights
        self.h = np.zeros(shape=(hidden_size, 1))                           # hidden state
        self.w_ih = kernel_initializer(shape=(hidden_size, input_size))     # input -> hidden
        self.w_hh = kernel_initializer(shape=(hidden_size, hidden_size))    # hidden -> hidden
        self.w_ho = kernel_initializer(shape=(output_size, hidden_size))    # hidden -> output

        self.b_h = bias_initializer(shape=(hidden_size, 1))                 # hidden bias
        self.b_o = bias_initializer(shape=(output_size, 1))                 # output bias

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)

        # update hidden state
        self.h = self.layer_activation(
            np.matmul(self.w_ih, out.T) + np.matmul(self.w_hh, self.h) + self.b_h)
        # rnn output
        act = np.matmul(self.w_ho, self.h) + self.b_o

        return np.clip(act.T, -1, 1)

    def get_weights(self):
        if self.use_bias:
            return [self.w_ih, self.w_hh, self.w_ho, self.b_h, self.b_o]
        return [self.w_ih, self.w_hh, self.w_ho]

    def set_weights(self, weights):
        if self.use_bias:
            self.w_ih, self.w_hh, self.w_ho, self.b_h, self.b_o = weights
        else:
            self.w_ih, self.w_hh, self.w_ho = weights
