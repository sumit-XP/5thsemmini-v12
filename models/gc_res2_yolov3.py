from typing import List, Tuple
import torch
from torch import nn
from .darknet_res2 import Darknet53Res2, conv_bn_lrelu
from .yolov3 import ConvSet, YOLODetection
from .gc_block import GCBlock


class GCRes2YOLOv3(nn.Module):
    """YOLOv3 with Darknet53-Res2 backbone, FPN, and GC blocks before each head.

    Returns outputs at scales [80x80, 40x40, 20x20] with channels anchors*(5+nc).
    """

    def __init__(self, num_classes: int, anchors_per_scale: int = 3) -> None:
        super().__init__()
        self.backbone = Darknet53Res2()
        self.anchors_per_scale = anchors_per_scale
        self.num_classes = num_classes

        # Top-down FPN and heads
        # P5 branch (from C5=1024)
        self.p5_convset = ConvSet(1024, 1024)
        self.p5_gc = GCBlock(512)  # after convset, feature will be 512
        self.p5_det = YOLODetection(512, anchors_per_scale, num_classes)

        # Reduce and upsample to build P4
        self.p5_to_p4 = conv_bn_lrelu(512, 256, 1, s=1, p=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_convset = ConvSet(256 + 512, 512)
        self.p4_gc = GCBlock(256)
        self.p4_det = YOLODetection(256, anchors_per_scale, num_classes)

        # Reduce and upsample to build P3
        self.p4_to_p3 = conv_bn_lrelu(256, 128, 1, s=1, p=0)
        self.p3_convset = ConvSet(128 + 256, 256)
        self.p3_gc = GCBlock(128)
        self.p3_det = YOLODetection(128, anchors_per_scale, num_classes)

        # Channel adapters after convset to match GC in_channels
        self.p5_csp = conv_bn_lrelu(512, 512, 1, s=1, p=0)
        self.p4_csp = conv_bn_lrelu(256, 256, 1, s=1, p=0)
        self.p3_csp = conv_bn_lrelu(128, 128, 1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3, c4, c5 = self.backbone(x)

        # P5
        p5 = self.p5_convset(c5)
        p5 = self.p5_csp(p5)
        p5_gc = self.p5_gc(p5)
        out_p5 = self.p5_det(p5_gc)  # 20x20

        # P4
        p4_in = torch.cat([self.upsample(self.p5_to_p4(p5)), c4], dim=1)
        p4 = self.p4_convset(p4_in)
        p4 = self.p4_csp(p4)
        p4_gc = self.p4_gc(p4)
        out_p4 = self.p4_det(p4_gc)  # 40x40

        # P3
        p3_in = torch.cat([self.upsample(self.p4_to_p3(p4)), c3], dim=1)
        p3 = self.p3_convset(p3_in)
        p3 = self.p3_csp(p3)
        p3_gc = self.p3_gc(p3)
        out_p3 = self.p3_det(p3_gc)  # 80x80

        # Return small->large order (80, 40, 20) similar to many training pipelines
        return [out_p3, out_p4, out_p5]
