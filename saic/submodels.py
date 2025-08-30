import torch
from torch import nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(
            self,
            channels,
            inverse=False,
            beta_min=1e-8,
            gamma_init=0.1
            ):
        """
        Simplified implementation of Generalized Divisive Normalization (GDN).

        Ballé, J., Laparra, V., & Simoncelli, E. P. (2016). Density modeling of
        images using a generalized normalization transformation. International
        Conference on Learning Representations (ICLR).

        NOTE this is the simplified implementation without alpha and epsilon.
        NOTE need to test whether this is the best or not.
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

        # unsqueeze g_amma from (C, C) into (C, C, 1, 1)
        # 1x1 conv across input channels with g_amma as weight and beta as bias
        norm = F.conv2d(x**2, weight=gamma_param.unsqueeze(2).unsqueeze(3), padding=0)
        norm = norm + beta_param.view(1, -1, 1, 1)

        if self.inverse:
            out = x * torch.sqrt(norm)
        else:
            out = x / torch.sqrt(norm)

        return out

class AnalysisTransform(nn.Module):
    def __init__(self, n, m):
        """
        Conv 5x5, N, stride 2
        GDN
        Conv 5x5, N, stride 2
        GDN
        Conv 5x5, N, stride 2
        GDN
        Conv 5x5, M, stride 2

        Padding=2 everywhere allows for output shape in this case to be 1/16th of input
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, n, (5, 5), stride=2, padding=2),
            GDN(n),
            nn.Conv2d(n, n, (5, 5), stride=2, padding=2),
            GDN(n),
            nn.Conv2d(n, n, (5, 5), stride=2, padding=2),
            GDN(n),
            nn.Conv2d(n, m, (5, 5), stride=2, padding=2),
        )
    
    def forward(self, x):
        out = self.model(x)
        return out
        

class SynthesisTransform(nn.Module):
    def __init__(self, n, m):
        """
        Deconv 5x5, N, stride 2
        IGDN
        Deconv 5x5, N, stride 2
        IGDN
        Deconv 5x5, N, stride 2
        IGDN
        Deconv 5x5, 3, stride 2
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(m, n, (5, 5), stride=2, padding=2, output_padding=1),
            GDN(n, inverse=True),
            nn.ConvTranspose2d(n, n, (5, 5), stride=2, padding=2, output_padding=1),
            GDN(n, inverse=True),
            nn.ConvTranspose2d(n, n, (5, 5), stride=2, padding=2, output_padding=1),
            GDN(n, inverse=True),
            nn.ConvTranspose2d(n, 3, (5, 5), stride=2, padding=2, output_padding=1), # assymetric synthesis into 3D image
        )

    def forward(self, x):
        out = self.model(x)
        return out

class HyperAnalysisTransform(nn.Module):
    def __init__(self, n, m):
        """
        Conv 3x3, N, stride 1
        Leaky ReLU
        Conv 5x5, N, stride 1
        Leaky ReLU
        Conv 5x5, N, stride 1
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(m, n, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(n, n, (5, 5), stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(n, n, (5, 5), stride=2, padding=2),
        )

    def forward(self, x):
        out = self.model(x)
        return out

class HyperSynthesisTransform(nn.Module):
    def __init__(self, n, m):
        """
        stride 2 in deconv introduces shape ambiguity that we resolve with output padding
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(n, n, (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(n, int(1.5 * n), (5, 5), stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(int(1.5 * n), 2*m, (3, 3), stride=1, padding=1),
        )
    
    def forward(self, x):
        out = self.model(x)
        return out

class Context(nn.Module):
    def __init__(self, m):
        """
        Context model used during parallel encoding pass and second decoding pass.

        The hyperprior suffices to decode the y_anchor bitstream, which is then
        passed to this model (alongside the hyperprior). The g_cm model is a
        simple convolution that will generate g_cm features for the parameter
        estimation model, to be used alongside the hyperprior, when estimating
        parameters for the non-anchor y_latent to be decoded.

        NOTE consider replacing this with self attention
        """
        super().__init__()
        self.conv = nn.Conv2d(m, 2*m, (5, 5), stride=1, padding='same')

    def forward(self, x):
        out = self.conv(x)
        return out

class ParameterEstimator(nn.Module):
    def __init__(self, n, m):
        """
        Takes concatenation of hyperprior and g_cm features to output
        parameters (mu, sigma) for latent distribution of y.

        Input shape: 2m from hyperprior and 2m from g_cm = 4m
        Output shape: m for mu and m for sigma = 2m
        """
        super().__init__()
        self.n = n
        self.m = m

        self.model = nn.Sequential(
            nn.Conv2d(2 * 2 * self.m, 640, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(640, 512, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 2 * self.m, kernel_size=1, stride=1),
        )
    
    def forward(self, x):
        params = self.model(x)
        mu, sigma = torch.chunk(params, 2, dim=1)

        # sigma has to be positive, we also scale for stability
        sigma = torch.exp(sigma * 0.5)
        return mu, sigma
