from typing import List, Tuple
import torch
from torch import nn


class ConvSet(nn.Module):
    """YOLOv3 conv set: [1x1, 3x3, 1x1, 3x3, 1x1]."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = out_ch // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, mid, 1, 1, 0, bias=False), nn.BatchNorm2d(mid), nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class YOLODetection(nn.Module):
    """Final detection conv producing anchors*(5+nc) outputs."""

    def __init__(self, in_ch: int, num_anchors: int, num_classes: int) -> None:
        super().__init__()
        self.pred = nn.Conv2d(in_ch, num_anchors * (5 + num_classes), 1, 1, 0)
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred(x)
