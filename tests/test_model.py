import pytest
import torch
from torch import nn

from pascal_voc_yolo.model import YOLOv1


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs)


def test_detector_produces_bounded_grid_predictions() -> None:
    model = YOLOv1(TinyBackbone(), backbone_channels=16)
    outputs = model(torch.randn(2, 3, 224, 224))
    assert outputs.shape == (2, 7, 7, 30)
    assert torch.all((0 <= outputs) & (outputs <= 1))


@pytest.mark.parametrize("shape", [(2, 1, 224, 224), (2, 3, 128, 128)])
def test_detector_rejects_wrong_image_shape(shape: tuple[int, ...]) -> None:
    model = YOLOv1(TinyBackbone(), backbone_channels=16)
    with pytest.raises(ValueError, match="224|RGB"):
        model(torch.randn(shape))
