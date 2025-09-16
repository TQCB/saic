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

    def _create_mask(self, y):
        # alternating 0 and 1 mask to decode (starting with 0 in top left)
        anchor_mask = torch.zeros_like(y)
        anchor_mask[:,::2,::2] = 1
        anchor_mask[:,1::2,1::2] = 1
        return anchor_mask
    
    def z_params_to_frequency(self, z_logits):
        pass

    def y_params_to_frequency(self, params):
        """
        Need to take params of shape mu, sigma, both (2*M) sized
        And return 
        """
        mu, sigma = params

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
        x = torch.cat((x, mask), 1)

        # get latents
        y = self.g_a(x)
        z = self.h_a(y)

        # get quantized
        y_hat = torch.round(y)
        z_hat = torch.round(z)

        # encode z_hat
        z_freqs = self.z_params_to_frequency(self.z_logits)
        z_encoder = RangeANSCoder(f=z_freqs)
        z_bitstream = z_encoder.stream_encode(z_hat)

        # get context features
        anchor_mask = self._create_mask(y_hat)
        y_half = y_hat * anchor_mask
        context_features = self.g_cm(y_half)

        # get hyperprior
        hyperprior = self.h_s(z_hat)

        # get entropy parameters to decode y
        features = torch.cat([hyperprior, context_features], dim=1)
        mu, sigma = self.g_ep(features)

        # separate latents and params into anchor and non-anchor sets
        non_anchor_mask = 1 - anchor_mask

        y_hat_anchors = y_hat * anchor_mask
        y_hat_non_anchors = y_hat * non_anchor_mask

        anchors_mu = mu * anchor_mask
        anchors_sigma = sigma * anchor_mask

        non_anchors_mu = mu * non_anchor_mask
        non_anchors_sigma = sigma * non_anchor_mask

        # create y encoders, encode anchor and non anchor latents
        y_anchor_freqs = self.y_params_to_frequency((anchors_mu, anchors_sigma))
        y_non_anchor_freqs = self.y_params_to_frequency((non_anchors_mu, non_anchors_sigma))

        y_anchor_encoder = RangeANSCoder(f=y_anchor_freqs)
        y_non_anchor_encoder = RangeANSCoder(f=y_non_anchor_freqs)

        y_anchor_bitstream = y_anchor_encoder(y_hat_anchors)
        y_non_anchor_bitstream = y_non_anchor_encoder(y_hat_non_anchors)

        return {
            "y_anchor_bitstream": y_anchor_bitstream,
            "y_non_anchor_bistream": y_non_anchor_bitstream,
            "z_bistream": z_bitstream,
        }
    
    def decompress(self, y_anchor_bitstream, y_nonanchor_bitstream, z_bitstream):
        """
        Inference decompression.
        """
        # --- Z HYPERPRIOR ---
        # Create z decoder from model parameters (z_logits)
        z_freqs = self.z_params_to_frequency(self.z_logits)
        z_decoder = RangeANSCoder(f=z_freqs)

        # Decode z_hat and create hyperprior
        z_hat = z_decoder.stream_decode(z_bitstream)
        hyperprior = self.h_s(z_hat)

        # --- Y ANCHORS ---
        # Calculate y_anchor params
        anchor_context = torch.zeros_like(hyperprior)
        anchor_features = torch.cat([hyperprior, anchor_context], dim=1)
        anchor_mu, anchor_sigma = self.g_ep(anchor_features)

        # Create y anchor decoder from params
        y_anchor_freqs = self.y_params_to_frequency((anchor_mu, anchor_sigma))
        y_anchor_decoder = RangeANSCoder(f=y_anchor_freqs)

        # Decode y hat anchors
        y_hat_anchors = y_anchor_decoder.stream_decode(y_anchor_bitstream)

        # --- Y NON ANCHORS ---
        # Calculate y non anchor parmas from decoded anchors
        non_anchor_context = self.g_cm(y_hat_anchors)
        non_anchor_features = torch.cat([hyperprior, non_anchor_context], dim=1)
        non_anchor_mu, non_anchor_sigma = self.g_ep(non_anchor_features)
        
        # Create y non anchor decoder from params
        y_non_anchor_freqs = self.y_params_to_frequency((non_anchor_mu, non_anchor_sigma))
        y_non_anchor_decoder = RangeANSCoder(f=y_non_anchor_freqs)

        # Decoded y hat non anchors
        y_hat_non_anchors = y_non_anchor_decoder.stream_decode(y_nonanchor_bitstream)

        # Final reconstruction
        y_hat = y_hat_anchors + y_hat_non_anchors
        x_hat = self.g_s(y_hat)

        return x_hat