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

class GatedRecurrentUnit(RecurrentNeuralNetwork):
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
        # import base rnn parameters
        from es.config import rnn_params
        self.layer_activation = rnn_params["layer_activation"]
        self.use_bias = rnn_params["use_bias"]
        hidden_size = rnn_params["hidden_size"]
        kernel_initializer = rnn_params["kernel_initializer"]
        bias_initializer = rnn_params["bias_initializer"]
        # import extra gru parameters
        from es.config import gru_params
        self.gate_activation = gru_params["gate_activation"]

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

class LongShortTermMemory(GatedRecurrentUnit):
    def __init__(self, input_size, output_size, every_t=True):
        self.every_t = every_t
        # import rnn base parameters
        from es.config import rnn_params
        self.layer_activation = rnn_params["layer_activation"]
        self.use_bias = rnn_params["use_bias"]
        kernel_initializer = rnn_params["kernel_initializer"]
        bias_initializer = rnn_params["bias_initializer"]
        # import extra gru parameters
        from es.config import gru_params
        self.gate_activation = gru_params["gate_activation"]
        # import extra lstm parameters
        from es.config import lstm_params
        self.cell_activation = lstm_params["cell_activation"]

        # initialize lstm parameters
        self.h = np.zeros(shape=(output_size, 1))                       # hidden state
        self.c = np.zeros(shape=(output_size, 1))                       # lstm hidden cell
        w_i = kernel_initializer(shape=(output_size, input_size))       # input -> hidden
        w_h = kernel_initializer(shape=(output_size, output_size))      # hidden -> hidden, output

        self.b = [bias_initializer(shape=(output_size, 1))] * 4         # biases

        self.w_f = [w_i, w_h]   # forget gate
        self.w_i = [w_i, w_h]   # input gate
        self.w_o = [w_i, w_h]   # output gate
        self.w_c = [w_i, w_h]   # lstm hidden cell
        
    def predict(self, inp):
        inp = np.expand_dims(inp.flatten(), 0)

        # forget gate
        f = self.gate_activation(
            np.matmul(self.w_f[0], inp.T) + np.matmul(self.w_f[1], self.h) + self.b[0])
        # input gate
        i = self.gate_activation(
            np.matmul(self.w_i[0], inp.T) + np.matmul(self.w_i[1], self.h) + self.b[1])
        # output gate
        o = self.gate_activation(
            np.matmul(self.w_o[0], inp.T) + np.matmul(self.w_o[1], self.h) + self.b[2])
        # hidden cell
        self.c = f * self.c + i * self.cell_activation(
            np.matmul(self.w_c[0], inp.T) + np.matmul(self.w_c[1], self.h) + self.b[3])
        # update hidden state
        self.h = o * self.layer_activation(self.c)

        return np.clip(o.T, -1, 1)

    def get_weights(self):
        if self.use_bias:
            return self.w_f + self.w_i + self.w_o + self.w_c + self.b
        return self.w_f + self.w_i + self.w_o + self.w_c

    def set_weights(self, weights):
        idx = 0
        # set forget gate weights
        len_f = len(self.w_f)
        self.w_f = weights[idx:idx+len_f]
        idx += len_f
        # set input gate weights
        len_i = len(self.w_i)
        self.w_i = weights[idx:idx+len_i]
        idx += len_i
        # set output gate weights
        len_o = len(self.w_o)
        self.w_o = weights[idx:idx+len_o]
        idx += len_o
        # set hidden cell weights
        len_c = len(self.w_c)
        self.w_c = weights[idx:idx+len_c]
        idx += len_c
        # set biases
        if self.use_bias:
            self.b = weights[idx:]
