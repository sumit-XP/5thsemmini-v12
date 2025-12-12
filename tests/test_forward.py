import torch
from project.models.gc_res2_yolov3 import GCRes2YOLOv3


def test_full_model_forward_pass():
    model = GCRes2YOLOv3(num_classes=4)
    x = torch.randn(2, 3, 640, 640)
    outs = model(x)
    assert isinstance(outs, list) and len(outs) == 3
    s80, s40, s20 = outs  # [N, 27, H, W]
    assert s80.shape[:2] == (2, 27) and s80.shape[-2:] == (80, 80)
    assert s40.shape[:2] == (2, 27) and s40.shape[-2:] == (40, 40)
    assert s20.shape[:2] == (2, 27) and s20.shape[-2:] == (20, 20)
