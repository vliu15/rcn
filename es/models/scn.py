"""
StructuredControlNet modeled after MultilayerPerceptron.
Original implementation detailed in this blog: https://medium.com/@mariosrouji/structured-control-nets-for-deep-reinforcement-learning-tutorial-icml-published-long-talk-paper-2ff99a73c8b
"""

import numpy as np
import copy


class StructuredControlNet(object):
    """
    Architecture:
    INPUT -> LINEAR
    INPUT -> MLP
    ----
    LINEAR + MLP -> OUTPUT
    """

    def __init__(self, input_size, output_size):
        # import mlp parameters:
        from es.config import mlp_params
        self.layer_activation = mlp_params["layer_activation"]
        layers = [input_size] + mlp_params["hidden_layers"] + [output_size]
        kernel_initializer = mlp_params["kernel_initializer"]
        # initialize mlp weights
        self.mlp_weights = []
        for i in range(len(layers) - 1):
            self.mlp_weights.append(kernel_initializer(
                shape=(layers[i], layers[i + 1])))

        # import linear parameters:
        from es.config import lin_params
        kernel_initializer = lin_params["kernel_initializer"]
        # initialize linear weights
        self.linear_weights = [kernel_initializer(shape=(input_size, 1))]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)

        # linear output:
        linear_out = np.dot(out, self.linear_weights[0])
        # mlp output:
        for layer in self.mlp_weights:
            out = np.dot(out, layer)
            out = self.layer_activation(out)
        # combine with sum:
        return np.clip(linear_out + out, -1, 1)

    def get_weights(self):
        tmp = copy.copy(self.mlp_weights)
        tmp.extend(self.linear_weights)
        return tmp

    def set_weights(self, weights):
        self.mlp_weights = weights[:len(self.mlp_weights)]
        self.linear_weights = weights[len(self.mlp_weights):]


class DeepStructuredControlNet(StructuredControlNet):
    """
    See StructuredControlNet for SCN architecture.
    Architecture:
    INPUT -> SCN -> ... -> SCN -> OUTPUT
    """

    def __init__(self, input_size, output_size):
        # import deep scn parameters
        from es.config import dscn_params
        self.use_residual = dscn_params["use_residual"]

        # initialize deep scn weights: list of SCN objects
        self.blocks = [StructuredControlNet(input_size, output_size)]
        for _ in range(dscn_params["num_blocks"] - 1):
            self.blocks.append(StructuredControlNet(output_size, output_size))

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        valid_shape = False

        for block in self.blocks:
            # linear output:
            lin_out = np.dot(out, block.linear_weights[0])
            # mlp output:
            mlp_out = out
            for layer in block.mlp_weights:
                mlp_out = np.dot(mlp_out, layer)
                mlp_out = block.layer_activation(mlp_out)
            # combine with sum:
            if self.use_residual and valid_shape:
                out += lin_out + mlp_out
            else:
                out = lin_out + mlp_out
            # residual does not apply to first iteration
            valid_shape = True

        return np.clip(out, -1, 1)

    def get_weights(self):
        weights = []
        for block in self.blocks:
            tmp = copy.copy(block.mlp_weights)
            tmp.extend(block.linear_weights)
            weights.extend(tmp)
        return weights

    def set_weights(self, weights):
        idx = 0
        for idx_block in range(len(self.blocks)):
            # extract mlp weights
            num_mlp = len(self.blocks[idx_block].mlp_weights)
            self.blocks[idx_block].mlp_weights = weights[idx:idx+num_mlp]
            idx += num_mlp
            # extract linear weights
            num_lin = len(self.blocks[idx_block].linear_weights)
            self.blocks[idx_block].linear_weights = weights[idx:idx+num_lin]
            idx += num_lin
