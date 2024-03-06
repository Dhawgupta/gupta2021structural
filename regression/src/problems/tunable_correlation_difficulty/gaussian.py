import numpy as np
import torch


class Gaussian:
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

    def __call__(self, batch_size):
        return batch(batch_size, self.mean, self.variance)


def batch(size, mean, variance):
    """
    Returns sample shaped (size, 1) for row-as-batch torch convention.
    """
    return (torch.randn(size) * np.sqrt(variance) + mean).reshape((size, 1))
