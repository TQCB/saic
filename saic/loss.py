import math
from typing import Sequence, Union

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from scipy.stats import norm

def gaussian(kernel_size: int, sigma: float) -> Tensor:
    """
    Compute 1D gaussian kernel

    Args
        kernel_size: length of kernel
        sigma: standard deviation of kernel
    """
    dist = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1)
    gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
    return (gauss / gauss.sum()).unsqueeze(dim=0)

def gaussian_kernel_2d(
        channel: int,
        kernel_size: Sequence[int],
        sigma: Sequence[float],
        ):
    """
    Compute 2D gaussian kernel

    Args
        channel: number of channels in image
        kernel_size: tuple for size of kernel (h, w)
        sigma: standrad deviation of kernel(s)
    """
    kernel_x = gaussian(kernel_size[0], sigma[0])
    kernel_y = gaussian(kernel_size[1], sigma[1])
    kernel = torch.matmul(kernel_x.t(), kernel_y)

    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1])

class SSIM(nn.Module):
    """
    Module implementing Structural Similarity Index Measure (SSIM).
    """
    def __init__(
            self,
            gaussian: bool=True,
            sigma: Union[float, Sequence[float]]=1.5,
            kernel_size: Union[int, Sequence[int]]=11,
            k1: float=0.01,
            k2: float=0.03,
            ):
        """
        Metric hyperparameters:
            gaussian = boolean whether or not to use gaussian or uniform kernel 
            sigma = 1.5 gaussian weighting variance
            kernel_size = 11 by default
            k1 = 0.01 by default
            k2 = 0.03 by default
            L = dynamic range of pixel values (2^n_bits_per_pixel - 1)
            c1 = (k1*L)^2
            c2 = (k2*L)^2
        """
        super().__init__()
        if not isinstance(sigma, Sequence):
            sigma = (sigma, sigma)

        if not isinstance(kernel_size, Sequence):
            kernel_size = (kernel_size, kernel_size)

        self.gaussian = gaussian
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.k1 = k1
        self.k2 = k2
    
    def forward(self, x, y):
        """

        """
        data_range = max(x.max() - x.min(), y.max() - y.min())
        c1 = pow(self.k1 * data_range, 2)
        c2 = pow(self.k2 * data_range, 2)

        channel = x.size(1)
        kernel = gaussian_kernel_2d(channel, self.kernel_size, self.sigma)
        kernel = kernel.to(device=x.device, dtype=x.dtype)

        conv_in = torch.cat((x, y, x*x, y*y, x*y)) # (5 * B, C, H, W)
        conv_out = F.conv2d(conv_in, kernel, groups=channel, padding='same')
        output_list = conv_out.split(x.shape[0])

        # m -> mu -> mean
        # s -> sigma -> sqrt(variance)
        # x, y -> pred, target
        # sq -> square

        # Means
        m_x_sq = output_list[0].pow(2)
        m_y_sq = output_list[1].pow(2)
        m_xy = output_list[0] * output_list[1]

        # Variances (should be non-negative)
        s_x_sq = torch.clamp(output_list[2] - m_x_sq, min=0.0)
        s_y_sq = torch.clamp(output_list[3] - m_y_sq, min=0.0)
        s_xy = output_list[4] - m_xy

        # Numerator and denominator of contrast sensitivity calculation!!!1
        # We keep these apart so that we can return them separately if we want
        upper = 2 * s_xy + c2
        lower = s_x_sq + s_y_sq + c2

        ssim_full_image = ((2 * m_xy + c1) * upper) / ((m_x_sq + m_y_sq + c1) * lower)

        return ssim_full_image

class SpatialRateDistortionLoss(nn.Module):
    """
    Module to combine rate and spatially weighted distortion losses with a
    lambda parameter.
    """
    def __init__(self,
                 lmbda: float=1e-2,
                 foreground_weight: float=10,
                 distortion_metric: str='mixed',
                 alpha: float=0.85,
                 ):
        super().__init__()
        if distortion_metric not in ["ssim", "mse", "mixed"]:
            raise ValueError(f"Distortion metric must be one of 'ssim', 'mse', or 'mixed', but got {distortion_metric}")
        
        self.lmbda = lmbda
        self.foreground_weight = foreground_weight
        self.distortion_metric = distortion_metric
        self.alpha = alpha
        
        self.ssim_loss = SSIM()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, output_dict, original_x, mask):
        # get num pixels for final bitrate
        num_pixels = original_x.size(0) * original_x.size(2) * original_x.size(3)
        x_hat = output_dict['x_hat']

        # distortion component
        weight_map = 1.0 + (self.foreground_weight - 1.0) * mask
        
        # Calculate distortion D based on the chosen metric
        if self.distortion_metric == "ssim":
            ssim_map = self.ssim_loss(x_hat, original_x)
            # Convert similarity to distortion (1 - SSIM) for minimization
            dssim_map = 1.0 - ssim_map
            weighted_distortion = weight_map * dssim_map
            self.D = torch.mean(weighted_distortion)
        
        elif self.distortion_metric == "mse":
            mse_map = self.mse_loss(x_hat, original_x)
            weighted_distortion = weight_map * mse_map
            self.D = torch.mean(weighted_distortion)

        elif self.distortion_metric == "mixed":
            # Calculate 1 - SSIM component
            ssim_map = self.ssim_loss(x_hat, original_x)
            dssim_map = 1.0 - ssim_map
            weighted_dssim = torch.mean(weight_map * dssim_map)
            
            # Calculate MSE component
            mse_map = self.mse_loss(x_hat, original_x)
            weighted_mse = torch.mean(weight_map * mse_map)
            
            # Combine them using alpha
            self.D = self.alpha * weighted_mse + (1.0 - self.alpha) * weighted_dssim
        
        # y rate component
        mu, sigma = output_dict['y_params']
        sigma = sigma.clamp(1e-6, 1e10)

        # use binned CDF probability
        gaussian = torch.distributions.Normal(mu, sigma)
        upper = gaussian.cdf(output_dict['y_hat'] + 0.5)
        lower = gaussian.cdf(output_dict['y_hat'] - 0.5)
        likelihood_y = (upper - lower).clamp(1e-6, 1.0)

        # z rate component
        z_logits = output_dict["z_params"]
        z_hat = output_dict["z_hat"].long()
        # z_hat_clamped = torch.clamp(z_hat, 0, z_logits.size(0) - 1)
        log_probs_z = F.log_softmax(z_logits, dim=0)
        likelihood_z = log_probs_z[z_hat]

        self.total_R_y = -torch.sum(torch.log2(likelihood_y))
        self.total_R_z = -torch.sum(likelihood_z) / math.log(2)

        self.R_bpp = (self.total_R_y + self.total_R_z) / num_pixels

        loss = self.lmbda * self.D + self.R_bpp
        return loss