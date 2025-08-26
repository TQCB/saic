from typing import Sequence

import numpy as np
from scipy.stats import norm

def parameters_to_frequency(
        alphabet: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        total_frequency: int=12
        ):
    """
    Given gaussian distribution parameters, return symbol frequencies over an
    alphabet.
    """
    # sigma = max(sigma, 1e-5) # avoid numerical issues from small sigma
    sigma = np.clip(sigma, a_min=1e-5, a_max=None)

    total_frequency = 2 ** total_frequency

    upper_bounds = norm.cdf(alphabet + 0.5, loc=mu, scale=sigma)
    lower_bounds = norm.cdf(alphabet - 0.5, loc=mu, scale=sigma)
    
    # probability of each symbol in alphabet based on integral
    probs = upper_bounds - lower_bounds

    # normalize probabilities
    probs_sum = np.sum(probs)
    if probs_sum == 0: # if 0, then use uniform distribution
        probs = np.ones_like(probs) / len(probs)
    else:
        probs /= probs_sum

    # get integer frequencies from probabilities
    frequencies = np.round(probs * total_frequency).astype(np.int64)
    frequencies[frequencies == 0] = 1 # all frequencies must be non-zero

    # correct error (offset of total freq)
    error = total_frequency - np.sum(frequencies)

    if error < 0:
        raise RuntimeError("Found integer frequencies are greater than total.")
    
    if error > 0:
        frequencies[np.argmax(probs)] += error

    return frequencies