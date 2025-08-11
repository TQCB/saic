"""
Compression model to be used in SAIC pipeline.
"""

import torch
from torch import nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(self, channels, inverse=False, beta_min=1e-8, gamma_init=0.1):
        """
        Simplified implementation of Generalized Divisive Normalization (GDN).

        Ballé, J., Laparra, V., & Simoncelli, E. P. (2016). Density modeling of
        images using a generalized normalization transformation. International
        Conference on Learning Representations (ICLR).


        """
        super().__init__()

        self.channels = channels
        self.inverse = inverse
        self.beta_min = beta_min

        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(torch.diag(torch.ones(channels) * gamma_init))

    def forward(self, x):
        # input x is shape (b, c, h, w)
        beta_param = torch.square(self.beta) + self.beta_min
        gamma_param = torch.square(self.gamma)

        # unsqueeze gamma from (C, C) into (C, C, 1, 1)
        # 1x1 conv across input channels with gamma as weight and beta as bias
        norm = F.conv2d(x**2, weight=gamma_param.unsqueeze(2).unsqueeze(3), padding=0)
        norm = norm + beta_param.view(1, -1, 1, 1)

        if self.inverse:
            out = x * torch.sqrt(norm)
        else:
            out = x / torch.sqrt(norm)

        return out

class AnalysisTransform(nn.Module):
    def __init__(self, N=128, M=256):
        """
        Conv 5x5, N, stride 2
        GDN
        Conv 5x5, N, stride 2
        GDN
        Conv 5x5, N, stride 2
        GDN
        Conv 5x5, M, stride 2
        """
        self.model = nn.Sequential(
            nn.Conv2d(4, N, (5, 5), stride=2),
            GDN(N),
            nn.Conv2d(N, N, (5, 5), stride=2),
            GDN(N),
            nn.Conv2d(N, N, (5, 5), stride=2),
            GDN(N),
            nn.Conv2d(N, M, (5, 5), stride=2),
        )
    
    def forward(self, x):
        out = self.model(x)
        return out
        

class SynthesisTransform(nn.Module):
    def __init__(self, N=128, M=256):
        """
        Deconv 5x5, N, stride 2
        IGDN
        Deconv 5x5, N, stride 2
        IGDN
        Deconv 5x5, N, stride 2
        IGDN
        Deconv 5x5, 3, stride 2
        """
        self.model = nn.Sequential(
            nn.ConvTranspose2d(M, N, (5, 5), stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, (5, 5), stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, (5, 5), stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 4, (5, 5), stride=2),
        )

    def forward(self, x):
        out = self.model(x)
        return out

class HyperAnalysisTransform(nn.Module):
    def __init__(self, N=128):
        """
        Conv 3x3, N, stride 1
        Leaky ReLU
        Conv 5x5, N, stride 1
        Leaky ReLU
        Conv 5x5, N, stride 1
        """
        self.model = nn.Sequential(
            nn.Conv2d(N, N, (3, 3), stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, (5, 5), stride=2),
        )

    def forward(self, x):
        out = self.model(x)
        return out

class HyperSynthesisTransform(nn.Module):
    def __init__(self, N=128, M=256):
        intermediate = int(1.5 * N)

        self.model = nn.Sequential(
            nn.Conv2d(N, N, (5, 5), stride=12),
            nn.LeakyReLU(),
            nn.Conv2d(N, intermediate, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(intermediate, 2*M, (3, 3), stride=1),
        )
    
    def forward(self, x):
        out = self.model(x)
        return out

class Context(nn.Module):
    def __init__(self):
        """
        """
        pass

    def forward(self, x):
        # rough outline of how this will work
        pass

class ParameterEstimator(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        """
        Conv 1x1, 640, stride 1
        Leaky ReLU
        Conv 1x1, 512, stride 1
        Leaky ReLU
        Conv 1x1, 2N, stride 1
        """

class Entropy(nn.Module):
    def __init__(self):
        pass

class HyperEntropy(nn.Module):
    def __init__(self):
        pass

class SAIC(nn.Module):
    def __init__(self):
        super().__init__()

        self.GA = AnalysisTransform()
        self.GS = SynthesisTransform()
        self.HA = HyperAnalysisTransform()
        self.HS = HyperSynthesisTransform()
        self.context = Context()
        self.parameter_estimator = ParameterEstimator()

        self.entropy = Entropy()
        self.hyper_entropy = HyperEntropy()

        self.rans_coder = None

    def forward(self, x, mask):
        """
        Forward pass to train model(s) to compress an image, conditioned by its
        mask, into 2 integer sequences (y, z).

        Args:
            x: torch.tensor
                shape: (N, C, H, W)
            mask: torch.tensor
                shape: (N, 1, H, W)
        """
        # alternating 0 and 1 mask to decode
        anchor_mask = torch.zeros_like(x)
        anchor_mask[:,::2,::2] = 1
        anchor_mask[:,1::2,1::2] = 1

        # Concatenate mask into 4th image channel
        # I will later replace this with a dedicated mask encoder and
        # mask conditioned quantization module
        x = torch.cat((x, mask), 1)

        # Forward through analysis transforms
        y = self.GA(x)
        z = self.HA(y)

        # Quantization during training -> addition of uniform noise ~ N(-0.5, 0.5)
        y_hat = y + torch.rand_like(y) - 0.5
        z_hat = z + torch.rand_like(z) - 0.5

        # Get parameters for PDF that entropy model will use, and to calculate rate loss
        """
        # context model takes in the quantized y_hat and the reconstructed z,

        # pseudocode to better understand this
        # on the first pass, we decode entropy parameters for anchor points (with 0 context)
        # we can now decode half the elements in y_hat using these parameters
        # on the second pass, we use the checkerboard model to generate a
        # context feature for non-anchors from the decoded anchors. We
        # concatenate this with our hyperprior and decode the remaining
        # latent y_hat

        def phi(z):
            y_anchor = []
            y_length = 1000
            for i in range(y_length):
                if y[i] in y_anchor:
                    out = G_ep(H_s(z_hat), 0)
                elif y[i] not in y_anchor:
                    out = G_ep(H_s(z_hat), G_cm(y_anchor, M * W)) # masked convolution parametrized by mask M and weights W
        """
        hyper_features = self.HS(z_hat) # no side information used to synthesize z
        y_half = y_hat * anchor_mask
        context_features = self.context(y_half)

        features = torch.cat(hyper_features, context_features)
        mu, sigma = self.parameter_estimator(features)

        # Reconstruct
        x_hat = self.GS(y_hat)
        
        # Return dictionary containing our reconstruction AND
        # y_hat, mu, sigma to be able to evaluate our rate loss
        out = {
            'x_hat': x_hat,
            'y_hat': y_hat,
            'mu': mu,
            'sigma': sigma,
        }

        return out
    
    def compress(self, x, mask):
        """
        Inference compression.

        Args:
            x: torch.tensor
                shape: (N, C, H, W)
            mask: torch.tensor
                shape: (N, 1, H, W)
        """
        pass
    
    def decompress(self, y_bitstream, z_bitstream):
        """
        Inference decompression.
        """
        pass