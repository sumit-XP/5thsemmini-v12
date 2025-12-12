import torch
from project.models.gc_block import GCBlock


def test_gcblock_output_shape():
    x = torch.randn(2, 128, 20, 20)
    gc = GCBlock(128, reduction=16)
    y = gc(x)
    assert y.shape == x.shape
