from typing import Sequence, Union

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

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
        Steps the forward needs to perform:

        1. Take image and split into patches
        2. Get m_x, m_y (means of x, y patches)
        3. Get s_x, s_y (variance of x, y patches)
        3. Get c_xy (covariance of x and y)
        4. calculate:
        numerator = (2 * m_x * m_y + c1) * (2 * c_xy + c2)
        denominator = (m_x^2 + m_y^2 + c1) * (s_x^2 + s_y^2 + c2)
        patch_metric = numerator/denominator

        5. average metric across all patches
        """
        data_range = max(x.max() - x.min(), y.max() - y.min())
        c1 = pow(self.k1 * data_range, 2)
        c2 = pow(self.k2 * data_range, 2)

        channel = x.size(1)
        kernel = gaussian_kernel_2d(channel, self.kernel_size, self.sigma)

        c1 = pow(self.k1 * data_range, 2)
        c2 = pow(self.k2 * data_range, 2)

        conv_in = torch.cat((x, y, x*x, y*y, x*y)) # (5 * B, C, H, W)
        conv_out = F.conv3d(conv_in, kernel, groups=channel)
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


class PSNRLoss(nn.Module):
    def __init__(self):
        raise NotImplementedError()

class RateLoss(nn.Module):
    def __init__(self):
        raise NotImplementedError()

class RateDistortionLoss(nn.Module):
    """
    Module to combine rate and distortion losses with a gamma parameter.
    """
    def __init__(self):
        super().__init__()