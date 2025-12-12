from typing import Tuple
import torch
from torch import nn


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + ReLU.

    Args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: conv kernel size
        stride: conv stride
        padding: conv padding
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Res2Block(nn.Module):
    """Res2Net-inspired block with 4 scale splits.

    Structure (scale=4):
    - Split input along channels into x1, x2, x3, x4 (C/4 each)
    - y1 = x1
    - y2 = Conv3x3(x2) [stride as provided]
    - y3 = Conv3x3(x3 + y2)
    - y4 = Conv3x3(x4 + y3)
    - concat = cat([y1, y2, y3, y4], dim=1)
    - out = Conv1x1(concat) to out_channels
    - Residual add (with 1x1 downsample if shape mismatch)

    Notes:
    - Uses BatchNorm + ReLU inside the 3x3/1x1 convs
    - Supports stride>1 for spatial downsampling
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        assert in_channels % 4 == 0, "in_channels must be divisible by 4 for 4-way split"
        self.stride = stride
        c = in_channels // 4

        # 3x3 transforms for groups 2..4; share stride for downsampling if needed
        self.conv2 = ConvBNAct(c, c, 3, stride=stride, padding=1)
        self.conv3 = ConvBNAct(c, c, 3, stride=1, padding=1)
        self.conv4 = ConvBNAct(c, c, 3, stride=1, padding=1)

        # Fuse back to desired out_channels
        self.fuse = ConvBNAct(in_channels, out_channels, 1, stride=1, padding=0)

        # Residual path if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into 4 equal channel groups
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        # Align spatial shapes for stride>1 across all paths
        if self.stride > 1:
            x1 = nn.functional.avg_pool2d(x1, kernel_size=self.stride, stride=self.stride)
            x3 = nn.functional.avg_pool2d(x3, kernel_size=self.stride, stride=self.stride)
            x4 = nn.functional.avg_pool2d(x4, kernel_size=self.stride, stride=self.stride)

        y1 = x1
        y2 = self.conv2(x2)
        y3 = self.conv3(x3 + y2)
        y4 = self.conv4(x4 + y3)

        concat = torch.cat([y1, y2, y3, y4], dim=1)
        out = self.fuse(concat)

        identity = self.downsample(x) if self.downsample is not None else x
        out = out + identity
        return out
