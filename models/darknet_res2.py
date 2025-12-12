from typing import List, Tuple
import torch
from torch import nn
from .res2_block import Res2Block, ConvBNAct


def conv_bn_lrelu(in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class Darknet53Res2(nn.Module):
    """Darknet-53-like backbone using Res2Blocks instead of Bottlenecks.

    Returns three feature maps for FPN: C3 (256), C4 (512), C5 (1024)
    Stage block counts follow YOLOv3: [1, 2, 8, 8, 4]
    """

    def __init__(self) -> None:
        super().__init__()
        # Stem
        self.stem = conv_bn_lrelu(3, 32, 3, s=1, p=1)

        # Stage 1
        self.down1 = conv_bn_lrelu(32, 64, 3, s=2, p=1)
        self.stage1 = self._make_stage(64, 64, num_blocks=1)

        # Stage 2
        self.down2 = conv_bn_lrelu(64, 128, 3, s=2, p=1)
        self.stage2 = self._make_stage(128, 128, num_blocks=2)

        # Stage 3
        self.down3 = conv_bn_lrelu(128, 256, 3, s=2, p=1)
        self.stage3 = self._make_stage(256, 256, num_blocks=8)

        # Stage 4
        self.down4 = conv_bn_lrelu(256, 512, 3, s=2, p=1)
        self.stage4 = self._make_stage(512, 512, num_blocks=8)

        # Stage 5
        self.down5 = conv_bn_lrelu(512, 1024, 3, s=2, p=1)
        self.stage5 = self._make_stage(1024, 1024, num_blocks=4)

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, num_blocks: int) -> nn.Sequential:
        blocks = []
        # First block can adjust channels if needed
        blocks.append(Res2Block(in_ch, out_ch, stride=1))
        for _ in range(1, num_blocks):
            blocks.append(Res2Block(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)

        x = self.down1(x)
        x = self.stage1(x)

        x = self.down2(x)
        x = self.stage2(x)

        x = self.down3(x)
        c3 = self.stage3(x)  # 1/8 resolution, 256 channels

        x = self.down4(c3)
        c4 = self.stage4(x)  # 1/16 resolution, 512 channels

        x = self.down5(c4)
        c5 = self.stage5(x)  # 1/32 resolution, 1024 channels

        return c3, c4, c5
