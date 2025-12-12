import torch
from project.models.res2_block import Res2Block


def test_res2block_output_shape():
    x = torch.randn(2, 64, 32, 32)
    blk = Res2Block(64, 64, stride=1)
    y = blk(x)
    assert y.shape == x.shape

    blk2 = Res2Block(64, 128, stride=2)
    y2 = blk2(x)
    assert y2.shape == (2, 128, 16, 16)
