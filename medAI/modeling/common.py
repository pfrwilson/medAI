import torch 
from torch import nn 
from einops import rearrange


class Patchify(torch.nn.Module):
    """
    Splits a tensor into patches of a given size.
    The input tensor is assumed to have shape (B, C, H, W).
    The output tensor will have shape (B, NH, NW, C, K1, K2), where
    K1 and K2 are the kernel sizes, and NH and NW are the number of patches
    in the height and width dimensions, respectively.

    Args:
        kernel_size (int or tuple): Size of the patch to be extracted.
        stride (int or tuple, optional): Stride of the patch to be extracted. Defaults to None.
        padding (int or tuple, optional): Padding of the patch to be extracted. Defaults to 0.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

    def n_windows(self, d, k, s, p): 
        return int((d - k + 2*p)/s + 1)

    def forward(self, x):
        c = x.shape[1]
        nh = self.n_windows(x.shape[-2], self.kernel_size[0], self.stride[0], self.padding[0])
        nw = self.n_windows(x.shape[-1], self.kernel_size[1], self.stride[1], self.padding[1])
        x = self.unfold(x)
        x = rearrange(x, 'b (c k1 k2) (nh nw) -> b nh nw c k1 k2', k1=self.kernel_size[0], k2=self.kernel_size[1], nh=nh, nw=nw, c=c)
        return x
    

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
