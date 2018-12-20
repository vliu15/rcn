import numpy as np
import copy

class RecurrentControlNet(object):
    """
    Nonlinear Module:
                   OUTPUT_n(t)
                      |
    HIDDEN(t-1) -> HIDDEN(t) -> ...
                      |
                   INPUT(t)
    
    Linear Module:
    INPUT(t) -> OUTPUT_l(t)

    Architecture:
    Nonlinear + Linear -> OUTPUT(t)
    """
    def __init__(self, input_size, output_size, every_t=True):
        self.every_t = every_t
        from config import rcn_params
        # import rcn parameters
        self.layer_activation = rcn_params["layer_activation"]
        self.n_use_bias = rcn_params["n_use_bias"]
        self.l_use_bias = rcn_params["l_use_bias"]
        hidden_size = rcn_params["hidden_size"]
        n_kernel_initializer = rcn_params["n_kernel_initializer"]
        l_kernel_initializer = rcn_params["l_kernel_initializer"]
        n_bias_initializer = rcn_params["n_bias_initializer"]
        l_bias_initializer = rcn_params["l_bias_initializer"]
        # initialize nonlinear module weights
        self.h = np.zeros(shape=(hidden_size, 1))                           # hidden state
        self.w_ih = n_kernel_initializer(shape=(hidden_size, input_size))     # input -> hidden
        self.w_hh = n_kernel_initializer(shape=(hidden_size, hidden_size))    # hidden -> hidden
        self.w_ho = n_kernel_initializer(shape=(output_size, hidden_size))    # hidden -> output
        self.b_h = n_bias_initializer(shape=(hidden_size, 1))                 # hidden bias
        self.b_o = n_bias_initializer(shape=(output_size, 1))                 # output bias
        # initialize linear module weights
        self.w_l = l_kernel_initializer(shape=(input_size, output_size))
        self.b_l = l_bias_initializer(shape=(output_size, 1))

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)

        # nonlinear module
        self.h = self.layer_activation(
            np.matmul(self.w_ih, out.T) + np.matmul(self.w_hh, self.h) + self.b_h)
        act = np.matmul(self.w_ho, self.h) + self.b_o
        n_out = act.T

        # linear module
        l_out = np.matmul(out, self.w_l) + self.b_l.T

        return np.clip(n_out + l_out, -1, 1)

    def get_weights(self):
        weights = [self.w_ih, self.w_hh, self.w_ho, self.w_l]
        if self.n_use_bias:
            weights.extend([self.b_h, self.b_o])
        if self.l_use_bias:
            weights.append(self.b_l)
        return weights

    def set_weights(self, weights):
        self.w_ih, self.w_hh, self.w_ho, self.w_l = weights[:4]
        idx = 4
        if self.n_use_bias:
           self.b_h, self.b_o = weights[idx:idx+2]
           idx += 2
        if self.l_use_bias:
            self.b_l = weights[idx]
