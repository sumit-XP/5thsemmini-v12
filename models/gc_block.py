from typing import Optional
import torch
from torch import nn


class GCBlock(nn.Module):
    """Global Context Block with LayerNorm and sigmoid attention.

    Steps:
    1) Attention map A = sigmoid(Conv1x1(x)) in R^{N,1,H,W}
    2) Normalized weights alpha = A / (sum(A) + eps) over spatial positions
    3) Context vector ctx = sum(alpha * x) in R^{N,C,1,1}
    4) Transform: C->C/r via 1x1, LayerNorm, ReLU, then expand C/r->C via 1x1
    5) Fuse: y = x + transform(ctx)

    Args:
        channels: number of input/output channels
        reduction: reduction ratio r for bottleneck (default 16)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        hidden = max(1, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        # LayerNorm over channel dimension; we'll apply on flattened (N, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.act = nn.ReLU(inplace=True)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # 1) attention
        a = self.attn_conv(x)  # (N,1,H,W)
        a = self.sigmoid(a)
        # 2) normalized weights
        weights_sum = a.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        alpha = a / weights_sum
        # 3) context aggregation
        ctx = (alpha * x).sum(dim=(2, 3), keepdim=True)  # (N,C,1,1)
        # 4) transform
        z = self.reduce(ctx)  # (N,hidden,1,1)
        z = z.view(n, -1)
        z = self.ln(z)
        z = self.act(z)
        z = z.view(n, -1, 1, 1)
        z = self.expand(z)
        # 5) fuse
        return x + z
