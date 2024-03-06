import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
from src.problems.tunable_correlation_difficulty import tunable_correlation_difficulty as tcd
from tqdm.notebook import tqdm

def target_function(domain_batch):
    # Normally one would see observation noise here, but the problem is hard
    # enough without it, so we don't introduce it as a confounding variable.
    return torch.sin(2 * np.pi * domain_batch ** 2)

def get_train(num_means, segment_length, difficulty, bound, batch_size):
    distributions = tcd.get_distributions(
        num_means = num_means,
        segment_length = segment_length,
        difficulty = difficulty,
        bound = bound,
    )
    domain_samples = np.expand_dims(np.zeros(len(distributions)*batch_size),axis=1)
    for i, distribution in enumerate(distributions):
        domain_samples[i*batch_size:(i+1)*batch_size] = distribution(batch_size)
    return domain_samples

def get_test(test_set_size, bound):
    return tcd.sample_overall_stationary(test_set_size, bound)