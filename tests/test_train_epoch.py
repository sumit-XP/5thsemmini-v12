import torch
from torch.utils.data import DataLoader, Dataset

from project.train import train_one_epoch
from project.models.gc_res2_yolov3 import GCRes2YOLOv3


class DummySet(Dataset):
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        x = torch.randn(3, 640, 640)
        y = torch.zeros((0, 4), dtype=torch.float32)
        return x, y


def test_train_one_epoch_cpu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl = DataLoader(DummySet(), batch_size=1, shuffle=False)
    model = GCRes2YOLOv3(num_classes=4).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    loss = train_one_epoch(model, dl, opt, scaler, device)
    assert isinstance(loss, float)
