import torch
from einops import rearrange, repeat
from torch import nn


class MaskPool(nn.Module):
    """Mask pooling layer for extracting features from a region of interest.

    This module receives a feature map and a mask and extracts and pools the 
    features within the masked region. The pooling operation can be either
    max or mean.

    Args:
        pool_type (str): Type of pooling to use. Options are 'max' and 'mean'.
    """
    def __init__(self, pool_type='max'):
        super(MaskPool, self).__init__()
        self.pool_type = pool_type

    def forward(self, x, mask):
        """Forward pass.

        Args:
            x (torch.Tensor): Input feature map with shape (B, C, H, W).
            mask (torch.Tensor): Binary mask with shape (B, 1, H, W).
        """
        B, C, H, W = x.shape 

        # cast mask to float and interpolate if necessary
        mask = mask.float()
        if mask.ndim == 3: 
            mask = mask.unsqueeze(1) # add channel dimension
        if mask.ndim == 4 and mask.shape[1] != 1: 
            raise ValueError(f'Invalid mask shape: {mask.shape}')
        if mask.shape[2:] != x.shape[2:]:
            mask = nn.functional.interpolate(mask, size=(H, W), mode='nearest')

        # cast back to boolean
        mask = (mask != 0)[:, 0, :, :]

        # apply mask to input
        mask = rearrange(mask, 'b h w -> (b h w)')
        x = rearrange(x, 'b c h w -> (b h w) c')
        batch_idx = torch.arange(B, device=x.device)
        batch_idx = repeat(batch_idx, 'b -> (b h w)', h=H, w=W)
        
        x = x[mask]
        batch_idx = batch_idx[mask]
        
        output = torch.zeros((B, C), device=x.device)
        for i in range(B):
            if (batch_idx == i).sum() == 0: 
                # no valid pixels in the mask - return zeros
                continue
            output[i] = x[batch_idx == i].mean(0) if self.pool_type == 'mean' else x[batch_idx == i].max(0).values

        return output
