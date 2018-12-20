"""
File for custom initializers.
"""
import numpy as np

def gaussian(mu=0, sigma=1):
    def f(shape):
        return sigma * np.random.randn(*shape) - mu
    return f