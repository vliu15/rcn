import numpy as np

def convolve(self, series, kernel, bias, stride, window):
    """
    Performs 1-D convolution along the time axis

    Args:
    :series: sequence of observations, shape=(timesteps, ob_shape)
    :kernel: convolution filters, shape=(out_channels, window, in_channels)
    :bias: biases for each kernel, shape=(out_channels)

    :stride: convolution stride
    :window: convolution window
    """
    i = 0
    conv_out = []
    while i + window <= series.shape[0]:
        inp = series[i:i+window]
        out = np.sum(np.sum(np.expand_dims(inp, axis=0) * kernel, axis=-1), axis=-1) + bias
        conv_out.append(out)
        i += stride
    return np.array(conv_out)