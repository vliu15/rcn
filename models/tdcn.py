import numpy as np

from utils.layers import convolve

class TimeDelayControlNet(object):
    """
    Nonlinear Module:
        TIME_SERIES: [INPUT(t-i), ..., INPUT(t-2), INPUT(t-1), INPUT(t)]
            where i is the length of the time series "memory"
            determined by window size and number of conv layers

        CONV_BLOCK: conv: in_channels -> out_channels + layer activation
            each block performs a 1-d convolution along the time axis
            and changes the number of channels (observation -> action shapes)

        TIME_SERIES -> CONV_BLOCK -> ... -> CONV_BLOCK -> OUTPUT_n

    Linear Module:
        INPUT(t) -> OUTPUT_l(t)

    Architecture:
        Nonlinear + Linear -> OUTPUT(t)
    """
    def __init__(self, input_size, output_size, every_t=True):
        self.every_t = every_t
        from config import tdcn_params
        # import tdcn parameters
        self.layer_activation = tdcn_params["layer_activation"]
        self.n_use_bias = tdcn_params["n_use_bias"]
        self.l_use_bias = tdcn_params["l_use_bias"]
        self.stride = tdcn_params["stride"]
        self.window = tdcn_params["window"]
        layers = tdcn_params["layers"]
        n_kernel_initializer = tdcn_params["n_kernel_initializer"]
        l_kernel_initializer = tdcn_params["l_kernel_initializer"]
        n_bias_initializer = tdcn_params["n_bias_initializer"]
        l_bias_initializer = tdcn_params["l_bias_initializer"]

        ## ==== NONLINEAR MODULE ==== ##
        # width: length of memory, determined from window sizes and number of hidden
        # layers so that after all convolutions, the output will be the action vector
        width = 1
        for _ in range(len(layers) + 1):
            width = (width - 1) * self.stride + self.window
        # initialize time series: sequence of observations
        self.series = np.zeros(shape=(width, input_size))
        # initialize convolution kernels
        inp_size = input_size
        layers.append(output_size)
        self.w_c, self.b_c = [], []
        # each kernel reduces length of time series (no padding) and changes
        # the size of information channels (i.e. observation size)
        for out_size in layers:
            self.w_c.append(n_kernel_initializer(shape=(out_size, self.window, inp_size)))
            self.b_c.append(n_bias_initializer(shape=(out_size)))
            inp_size = out_size

        ## ==== LINEAR MODULE ==== ##
        self.w_l = l_kernel_initializer(shape=(input_size, output_size))
        self.b_l = l_bias_initializer(shape=(output_size, 1))

    def predict(self, inp):
        inp = inp.flatten()

        ## ==== NONLINEAR MODULE ==== ##
        # update time series with new observation
        ser = np.copy(self.series)
        ser[:-1] = np.copy(self.series[1:])
        ser[-1] = inp
        self.series = ser
        # convolve as time delay
        conv_out = ser
        for k, b in zip(self.w_c, self.b_c):
            # output of convolution is a shortened time length and hidden
            # size of output channels
            conv_out = convolve(conv_out, k, b, self.stride, self.window)
            conv_out = self.layer_activation(conv_out)
        # initial width should be perfectly reduced to a time series of length 1
        n_out = conv_out

        ## ==== LINEAR MODULE ==== ##
        l_out = np.matmul(np.expand_dims(inp, axis=0), self.w_l) + self.b_l.T

        return np.clip(n_out + l_out, -1, 1)

    def get_weights(self):
        weights = self.w_c + [self.w_l]
        if self.n_use_bias:
            weights = weights + self.b_c
        if self.l_use_bias:
            weights = weights + [self.b_l]
        return weights

    def set_weights(self, weights):
        self.w_c = weights[:len(self.w_c)]
        self.w_l = weights[len(self.w_c)]
        if self.n_use_bias:
            self.b_c = weights[len(self.w_c)+1:len(self.w_c)+1+len(self.b_c)]
        if self.l_use_bias:
            self.b_l = weights[-1]