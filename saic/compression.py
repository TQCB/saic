"""
Compression model to be used in SAIC pipeline.
"""

import torch
from torch import nn

import submodels as sm

class HyperpriorCheckerboardCompressor(nn.Module):
    def __init__(
            self,
            n: int,
            m: int,
            z_alphabet_size: int,
            ):
        super().__init__()

        self.g_a = sm.AnalysisTransform(n=n, m=m)
        self.g_s = sm.SynthesisTransform(n=n, m=m)
        self.h_a = sm.HyperAnalysisTransform(n=n, m=m)
        self.h_s = sm.HyperSynthesisTransform(n=n, m=m)
        self.g_cm = sm.Context(m=m)
        self.g_ep = sm.ParameterEstimator(n=n, m=m)

        self.z_alphabet_size = z_alphabet_size
        self.z_logits = nn.Parameter(torch.rand(self.z_alphabet_size))

        self.rans_coder = None

    def _create_mask(self, y):
        # alternating 0 and 1 mask to decode (starting with 0 in top left)
        anchor_mask = torch.zeros_like(y)
        anchor_mask[:,::2,::2] = 1
        anchor_mask[:,1::2,1::2] = 1
        return anchor_mask

    def forward(self, x, mask):
        """
        Forward pass to train model(s). Doesn't actually interact with the rANS coder.

        Args:
            x: torch.tensor
                shape: (N, C, H, W)
            mask: torch.tensor
                shape: (N, 1, H, W)
        """
        # Concatenate mask into 4th image channel
        # I will later replace this with a dedicated mask encoder and
        # mask conditioned quantization module
        x = torch.cat((x, mask), 1)

        # Forward through analysis transforms
        y = self.g_a(x)
        z = self.h_a(y)

        anchor_mask = self._create_mask(y)

        # Quantization during training -> addition of uniform noise ~ N(-0.5, 0.5)
        y_hat = y + torch.rand_like(y) - 0.5
        z_hat = z + torch.rand_like(z) - 0.5

        # Get parameters for PDF that entropy model will use, and to calculate rate loss
        hyperprior = self.h_s(z_hat) # no side information used to synthesize z
        y_half = y_hat * anchor_mask
        context_features = self.g_cm(y_half)

        features = torch.cat([hyperprior, context_features], dim=1)
        mu, sigma = self.g_ep(features)

        # Reconstruct
        x_hat = self.g_s(y_hat)
        
        # Return dictionary containing:
        # - all our latents, and our reconstruction
        # - all latent parameters
        # Using these, we can evaluate the distortion of our reconstruction,
        # and we can evaluate how well our latent parameters represent our
        # quantized latents
        out = {
            'x_hat': x_hat,
            'y_hat': y_hat,
            'z_hat': z_hat,
            'y_params': (mu, sigma),
            'z_params': self.z_logits,
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
    
    def decompress(self, y_anchor_bitstream, y_nonanchor_bitstream, z_bitstream):
        """
        Inference decompression.
        """
        pass