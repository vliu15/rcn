"""
Fourier Series models, built of vanilla StructuredControlNet.
Original implementation detailed in this blog: https://medium.com/@mariosrouji/structured-control-nets-for-deep-reinforcement-learning-tutorial-icml-published-long-talk-paper-2ff99a73c8b
"""

import numpy as np
import copy


class LocomotorNet(object):
    def __init__(self, input_size, output_size):
        # import frs parameters:
        from es.config import frs_params
        a_init = frs_params["amp_initializer"]
        f_init = frs_params["freq_initializer"]
        p_init = frs_params["phase_initializer"]
        self.frs_func = frs_params["frs_func"]
        self.frs_size = frs_params["frs_size"]
        # initialize frs weights
        self.frs_weights = []
        for i in range(frs_params["frs_size"]):
            self.frs_weights.append(
                (a_init(output_size), f_init(output_size), p_init(output_size)))

        # import linear parameters:
        from es.config import lin_params
        kernel_initializer = lin_params["kernel_initializer"]
        # initialize linear weights
        self.linear_weights = [kernel_initializer(
            shape=(input_size, output_size))]

    def predict(self, inp, t):
        out = np.expand_dims(inp.flatten(), 0)

        # linear output:
        linear_out = np.dot(out, self.linear_weights[0])
        # frs output:
        frs_out = np.zeros_like(linear_out)
        for s in self.frs_weights:
            frs_out = frs_out + s[0] * self.frs_func(s[1] * t - s[2])
        # combine with sum:
        return np.clip(linear_out + frs_out, -1, 1)

    def get_weights(self):
        weights = []
        for s in self.frs_weights:
            tmp = copy.copy(s)
            weights.extend(tmp)
        weights.extend(self.linear_weights)
        return weights

    def set_weights(self, weights):
        idx = 0
        for idx_s in range(len(self.frs_weights)):
            num_frs = len(self.frs_weights[idx_s])
            self.frs_weights[idx_s] = weights[idx:idx+num_frs]
            idx += num_frs
        self.linear_weights = weights[idx:]
