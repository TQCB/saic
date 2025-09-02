from typing import Sequence

import torch
from torchvision import transforms
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

def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    Original normalization is:
    `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

    So the inverse is:
    (image * std) + mean

    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    """
    inv_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    inv_image = inv_norm(image)

    return torch.clamp(inv_image, 0, 1)

def convert_tensor_to_image(tensor):
    """
    Converts a PyTorch tensor into a displayable NumPy image array.
    Assumes the input tensor is in the range [0, 1].
    """
    image = tensor.cpu().detach()
    image = denormalize_image(image)
    image = image.permute(1, 2, 0)
    image = torch.clamp(image, 0, 1)
    image = (image * 255).to(torch.uint8)

    return image.numpy()

def plot_progress(path, epoch, steps, output_dict, image, bitrate):
    """
    Plot original image, reconstructed image, and bit rates
    """
    original_img = convert_tensor_to_image(image[0])
    reconstructed_img = convert_tensor_to_image(output_dict['x_hat'][0])

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Image(z=original_img), row=1, col=1)
    fig.add_trace(go.Image(z=reconstructed_img), row=1, col=2)

    fig.update_layout(title=dict(text=f"Epoch: {epoch}, Steps:{steps}<br>BPP: {bitrate:.6f}"))
    fig.write_image(path + f"/progress_{epoch}_{steps}.jpg")