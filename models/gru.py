import numpy as np

class GatedRecurrentUnit(object):
    """
    Architecture:
                   OUTPUT(t)
                      |
    HIDDEN(t-1) -> HIDDEN(t) -[gate]-> ...
                      |
                   INPUT(t)
    """
    def __init__(self, input_size, output_size, every_t=True):
        self.every_t = every_t
        # import base gru parameters
        from config import gru_params
        self.layer_activation = gru_params["layer_activation"]
        self.gate_activation = gru_params["gate_activation"]
        self.use_bias = gru_params["use_bias"]
        hidden_size = gru_params["hidden_size"]
        kernel_initializer = gru_params["kernel_initializer"]
        bias_initializer = gru_params["bias_initializer"]
        
        # initialize gru weights
        self.h = np.zeros(shape=(hidden_size, 1))                        # hidden state
        w_ih = kernel_initializer(shape=(hidden_size, input_size))       # input -> hidden
        w_hh = kernel_initializer(shape=(hidden_size, hidden_size))      # hidden -> hidden
        self.w_ho = kernel_initializer(shape=(output_size, hidden_size)) # hidden -> output

        self.b = [bias_initializer(shape=(hidden_size, 1))] * 3 + \
            [bias_initializer(shape=(output_size, 1))]                 # biases

        self.w_u = [w_ih, w_hh]     # update gate weights
        self.w_r = [w_ih, w_hh]     # reset gate weights
        self.w_h = [w_ih, w_hh]     # weights to update hidden state

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)

        # gate from input
        update = self.gate_activation(
            np.matmul(self.w_u[0], out.T) + np.matmul(self.w_u[1], self.h) + self.b[0])
        # vector to weight hidden state (to add with input)
        reset = self.gate_activation(
            np.matmul(self.w_r[0], out.T) + np.matmul(self.w_r[1], self.h) + self.b[1])
        # update hidden state
        self.h = update * self.h + (1 - update) * self.layer_activation(
            np.matmul(self.w_h[0], out.T) + np.matmul(self.w_h[1], reset * self.h) + self.b[2])
        # gru output
        act = np.matmul(self.w_ho, self.h) + self.b[-1]

        return np.clip(act.T, -1, 1)

    def get_weights(self):
        if self.use_bias:
            return self.w_u + self.w_r + self.w_h + [self.w_ho] + self.b
        return self.w_u + self.w_r + self.w_h + [self.w_ho]

    def set_weights(self, weights):
        idx = 0
        # set update gate weights
        len_u = len(self.w_u)
        self.w_u = weights[idx:idx+len_u]
        idx += len_u
        # set reset gate weights
        len_r = len(self.w_r)
        self.w_r = weights[idx:idx+len_r]
        idx += len_r
        # set hidden state update weights
        len_h = len(self.w_h)
        self.w_h = weights[idx:idx+len_h]
        idx += len_h
        # set output weight
        self.w_ho = weights[idx]
        if self.use_bias:
            self.b = weights[idx+1:]