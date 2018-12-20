"""
File for various layer activations.
"""

import numpy as np


def sin(x, amp=1, freq=1, phase=0):
    return amp * np.sin(freq * x - phase)


def sinc(x, amp=1, freq=1, phase=0):
    return amp * np.sin(freq * x - phase) / x if x != 0 else amp


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sinc_act(x):
    return 1 * np.sin(1 * x - 0) / x


def linear(x):
    return x


def relu(x):
    x[x < 0] = 0
    return x
