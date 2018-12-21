import numpy as np

def convolve(series, kernel, bias, stride, window):
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
    # compute scalar output of dot product at each window
    while i + window <= series.shape[0]:
        # take current window
        inp = series[i:i+window]
        # find dot product of kernel and time series window + bias
        out = np.sum(np.sum(np.expand_dims(inp, axis=0) * kernel, axis=-1), axis=-1) + bias
        # append scalar to list to cast to array later
        conv_out.append(out)
        i += stride
    return np.array(conv_out)