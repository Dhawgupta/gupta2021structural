import numpy as np

import torch

from src.problems.tunable_correlation_difficulty import gaussian as gaussian


def necessary_attraction_coefficient(noise_variance, stationary_variance):
    if noise_variance == 0 and stationary_variance == 0:
        return 0
    else:
        return 1 - np.sqrt(1 - noise_variance / stationary_variance)


def walk_step(previous_walk_step, noise_variance, attraction_coefficient):
    """
    Follows recurrence S_t = (1-c)S_{t-1} + N(0, sigma^2).
    Note that noise_variance = sigma^2
    Takes a float, returns a float.  Does not need to be vectorized, because computing the walk is not a bottleneck.
    """
    return (1 - attraction_coefficient) * previous_walk_step + torch.randn(1).item() * np.sqrt(noise_variance)


def walk_from(start, noise_variance, attraction_coefficient, steps):
    walk = [start]
    for t in range(steps - 1):
        walk.append(walk_step(walk[t], noise_variance, attraction_coefficient))
    return walk


def walk_stationary_variance(noise_variance, attraction_coefficient):
    if noise_variance > 0:
        return noise_variance / (2 * attraction_coefficient - attraction_coefficient ** 2)
    else:
        return 0


def sample_walk_stationary_dist(size, noise_variance, attraction_coefficient):
    """
    Returns sample shaped (size, 1) for row-as-batch torch convention.
    """
    return gaussian.batch(size, 0, walk_stationary_variance(noise_variance, attraction_coefficient))


def walk_from_stationary_dist(noise_variance, attraction_coefficient, steps):
    start_position = sample_walk_stationary_dist(1, noise_variance, attraction_coefficient).item()

    return walk_from(start_position, noise_variance, attraction_coefficient, steps)
