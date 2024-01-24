# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from typing import List, Optional

import math
import torch
from torch import nn
from torch import Tensor


class GaussianFourierFeatureTransform(nn.Module):
    """
    https://github.com/ndahlquist/pytorch-fourier-feature-networks
    """
    def __init__(
        self,
        input_channel: int = 1,
        n_fourier_feats: int = 512,
        scales: List[int] = [0.01, 0.1, 1, 5, 10, 20, 50, 100],
        include_original: bool = False,
        ):
        super().__init__()
        self.input_channel = input_channel
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.include_original = include_original
        
        n_scale_feats = n_fourier_feats // (2 * len(scales))
        assert n_scale_feats * 2 * len(scales) == n_fourier_feats, \
            f"n_fourier_feats: {n_fourier_feats} must be divisible by 2 * len(scales) = {2 * len(scales)}"
            
        assert input_channel == 1, \
            f"Expected input_channel to be 1, got {input_channel}"
            
        B_size = (1, n_scale_feats, 1, 1)
        B_omegas = torch.cat([torch.randn(B_size) * scale for scale in scales], dim=1)
        self.register_buffer('B_omegas', B_omegas)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() >= 4, f"Expected 3 or more dimensional input (got {x.dim()}D input)"
        batch_size, channel, height, width = x.shape

        assert channel == 1, \
            f"Expected input to have 1 channel for now (got {channel} channels)"

        # x = torch.einsum('... t n, n d -> ... t d', [x, self.B])
        x_omega = x * self.B_omegas # [bz, 1, height, width] * [1, n_fourier_feats, 1, 1]
        x_omega = 2 * math.pi * x_omega
        
        if self.include_original:
            return torch.cat([torch.sin(x_omega), torch.cos(x_omega)] + 8*[x], dim=1)
        
        return torch.cat([torch.sin(x_omega), torch.cos(x_omega)], dim=1)