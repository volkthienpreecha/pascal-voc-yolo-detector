import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pascal_voc_yolo.engine import evaluate, resolve_device, train_epoch


class TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(13))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.bias).reshape(1, 1, 1, 13).expand(images.shape[0], -1, -1, -1)


def test_engine_runs_one_synthetic_batch() -> None:
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 4, 4), torch.zeros(4, 1, 1, 13)), batch_size=2)
    model = TinyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    training = train_epoch(model, loader, optimizer, torch.device("cpu"), boxes_per_cell=2)
    validation = evaluate(model, loader, torch.device("cpu"), boxes_per_cell=2)
    assert training["total"] >= 0
    assert validation["confidence"] >= 0


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
