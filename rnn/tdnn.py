import numpy as np
import copy

from utils.layers import convolve

class TimeDelayNeuralNetwork(object):
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
        from config import tdnn_params
        # import tdnn parameters
        self.layer_activation = tdnn_params["layer_activation"]
        self.use_bias = tdnn_params["use_bias"]
        self.stride = tdnn_params["stride"]
        self.window = tdnn_params["window"]
        layers = tdnn_params["layers"]
        kernel_initializer = tdnn_params["kernel_initializer"]
        bias_initializer = tdnn_params["bias_initializer"]
        # initialize time series
        width = 1
        for _ in range(len(layers) + 1):
            width = (width - 1) * self.stride + self.window
        self.series = np.zeros(shape=(width, input_size))
        # initialize convolution kernels
        inp_size = input_size
        layers.append(output_size)
        self.kernels, self.biases = [], []
        for out_size in layers:
            self.kernels.append(kernel_initializer(shape=(out_size, self.window, inp_size)))
            self.biases.append(bias_initializer(shape=(out_size)))
            inp_size = out_size

    def predict(self, inp):
        # adjust time series
        ser = np.copy(self.series)
        ser[:-1] = np.copy(self.series[1:])
        ser[-1] = inp.flatten()
        self.series = ser

        # convolve as time delay
        conv_out = ser
        for k, b in zip(self.kernels, self.biases):
            conv_out = convolve(conv_out, k, b, self.stride, self.window)
            conv_out = self.layer_activation(conv_out)
        
        return conv_out


    def get_weights(self):
        if self.use_bias:
            return self.kernels + self.biases
        return self.kernels

    def set_weights(self, weights):
        if self.use_bias:
            l = len(weights)
            self.kernels = weights[:l//2]
            self.biases = weights[l//2:]
        else:
            self.kernels = weights