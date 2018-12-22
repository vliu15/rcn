import numpy as np

class LongShortTermMemory(object):
    def __init__(self, input_size, output_size, every_t=True):
        self.every_t = every_t
        # import lstm parameters
        from config import lstm_params
        self.layer_activation = lstm_params["layer_activation"]
        self.gate_activation = lstm_params["gate_activation"]
        self.cell_activation = lstm_params["cell_activation"]
        self.use_bias = lstm_params["use_bias"]
        kernel_initializer = lstm_params["kernel_initializer"]
        bias_initializer = lstm_params["bias_initializer"]

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

        return np.clip(self.h.T, -1, 1)

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