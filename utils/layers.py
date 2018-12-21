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
    # computes dot product between each filter and series window
    def kernel_dot(inp):
        out = np.sum(np.sum(np.expand_dims(inp, axis=0) * kernel, axis=-1), axis=-1) + bias
        return out

    ## ========  USE FOR LOOP  ======== ##
    # i = 0
    # conv_out = []
    # while i + window <= series.shape[0]:
    #     # take current window
    #     inp = series[i:i+window]
    #     # find dot product of kernel and time series window + bias
    #     out = kernel_dot(inp)
    #     # append scalar to list to cast to array later
    #     conv_out.append(out)
    #     i += stride
    # return np.array(conv_out)

    ## ======== USE PYTHON MAP ======== ##
    # inputs = [series[i:i+window] for i in range(series.shape[0]-window+1)]
    # conv_out = map(kernel_dot, inputs)
    # return np.array(list(outputs))

    ## ======== USE LIST COMP  ======== ##
    conv_out = [kernel_dot(series[i:i+window]) for i in range(series.shape[0]-window+1)]
    return np.array(conv_out)
