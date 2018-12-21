"""
File for custom initializers.
"""
import numpy as np

def gaussian(mu=0, sigma=1e-3):
    """
    Returns Gaussian initializer from distribution ~ N(mu, sigma^2)
    """
    def f(shape):
        return sigma * np.random.randn(*shape) + mu
    return f

def uniform(scale=1e-3):
    """
    Returns uniform initializer from distribution ~ U[0, scale)
    """
    def f(shape):
        return scale * np.random.uniform(size=shape)
    return f

def constant(fill=1e-3):
    """
    Returns constant initializer with value fill
    """
    def f(shape):
        return fill * np.ones(shape=shape)
    return f