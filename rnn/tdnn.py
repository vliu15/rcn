import numpy as np
import copy

from utils.layers import convolve

class TimeDelayNeuralNetwork(object):
    """
    Architecture:
    
    TIME_SERIES: [o(t-i), ..., o(t-3), o(t-2), o(t-1), o(t)]
        where i is the length of the time series "memory"
        determined by window size and number of conv layers

    CONV_BLOCK: conv: in_channels -> out_channels + layer activation
        each block performs a 1-d convolution along the time axis
        and changes the number of channels (observation -> action shapes)

    TIME_SERIES -> CONV_BLOCK -> ... -> CONV_BLOCK -> OUTPUT
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

        # width: length of memory
        # width is determined from window sizes and number of hidden layers
        # so that after all convolutions, the output will be the action vector
        width = 1
        for _ in range(len(layers) + 1):
            width = (width - 1) * self.stride + self.window
        # initialize time series
        # stores sequence of observations
        self.series = np.zeros(shape=(width, input_size))

        # initialize convolution kernels
        inp_size = input_size
        layers.append(output_size)
        self.kernels, self.biases = [], []
        # each kernel reduces length of time series (no padding) and changes
        # the size of information channels (i.e. observation size)
        for out_size in layers:
            self.kernels.append(kernel_initializer(shape=(out_size, self.window, inp_size)))
            self.biases.append(bias_initializer(shape=(out_size)))
            inp_size = out_size

    def predict(self, inp):
        # update time series with new observation
        ser = np.copy(self.series)
        ser[:-1] = np.copy(self.series[1:])
        ser[-1] = inp.flatten()
        self.series = ser

        # convolve as time delay
        conv_out = ser
        for k, b in zip(self.kernels, self.biases):
            # output of convolution is a shortened time length and hidden
            # size of output channels
            conv_out = convolve(conv_out, k, b, self.stride, self.window)
            conv_out = self.layer_activation(conv_out)
        
        # initial width should be perfectly reduced to a time series of length 1
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