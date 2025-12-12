import torch
from project.utils.loss import ciou_loss


def test_ciou_basic_properties():
    a = torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32)
    b = torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32)
    loss_same = ciou_loss(a, b)
    assert torch.allclose(loss_same, torch.zeros_like(loss_same), atol=1e-5)

    a2 = torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
    loss_diff = ciou_loss(a2, b)
    assert (loss_diff > 0).all()
