from typing import Sequence

import numpy as np
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

def plot_progress(output_dict, image, y_rate, z_rate):
    """
    Plot original image, reconstructed image, and bit rates
    """
    image = image[0].detach()
    image = image.permute(1, 2, 0)
    reconstructed = output_dict['x_hat'][0].detach()
    reconstructed = reconstructed.permute(1, 2, 0)

    print(image.shape)
    print(reconstructed.shape)

    total_rate = y_rate + z_rate
    bitrate = total_rate / image.numel()

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Image(z=image), row=1, col=1)
    fig.add_trace(go.Image(z=reconstructed), row=1, col=2)

    fig.update_layout(title=dict(text=f"y_rate: {y_rate}\nz_rate: {z_rate}\ntotal_rate: {total_rate}\nbitrate: {bitrate}"))
    fig.show()