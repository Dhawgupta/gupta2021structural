import torch

from itertools import zip_longest

from src.problems.tunable_correlation_difficulty.gaussian import Gaussian, batch
from src.problems.tunable_correlation_difficulty.walk import walk_from_stationary_dist, necessary_attraction_coefficient


def segmenter(iterable, num_segments):
    args = [iter(iterable)] * num_segments
    return list(zip_longest(*args))


def get_distributions(num_means, segment_length, difficulty, bound):
    noise_var = walk_noise_variance(difficulty, bound)
    attraction_coeff = necessary_attraction_coefficient(noise_var, walk_stationary_variance(difficulty, bound))
    means = walk_from_stationary_dist(noise_var, attraction_coeff, num_means)

    segment_var = iid_segment_variance(difficulty, bound)
    distributions = []
    for mean in means:
        for _step in range(segment_length):
            distributions.append(Gaussian(mean, segment_var))
    return distributions


def overall_stationary_variance(bound):
    # This could be done more generally by passing a bound and probability, and using the
    # Gaussian CDF to get the variance guaranteeing P(|x| < b) = probability.
    # For now we pick ~95% probability, so 2 standard deviations.
    return (bound / 2) ** 2


def iid_segment_variance(difficulty, bound):
    return (1 - difficulty) * overall_stationary_variance(bound)


def walk_stationary_variance(difficulty, bound):
    return difficulty * overall_stationary_variance(bound)


def walk_noise_variance(difficulty, bound):
    return difficulty * walk_stationary_variance(difficulty, bound)


def sample_overall_stationary(sample_size, bound):
    return batch(sample_size, 0, overall_stationary_variance(bound))


def grid_covering_stationary_support(test_size, bound):
    return torch.linspace(-1, 1, test_size).reshape((test_size, 1))
