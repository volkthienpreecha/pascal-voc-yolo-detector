"""YOLOv1-style detector head with an injectable feature backbone."""

import torch
from torch import nn


class YOLOv1(nn.Module):
    """Predict grid-aligned boxes and class probabilities from feature maps."""

    def __init__(
        self,
        backbone: nn.Module,
        backbone_channels: int,
        grid_size: int = 7,
        boxes_per_cell: int = 2,
        num_classes: int = 20,
    ) -> None:
        super().__init__()
        if min(backbone_channels, grid_size, boxes_per_cell, num_classes) <= 0:
            raise ValueError("model dimensions must be positive")
        self.backbone = backbone
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        output_width = 5 * boxes_per_cell + num_classes
        head_channels = max(32, min(256, backbone_channels))
        self.features = nn.Sequential(
            nn.Conv2d(backbone_channels, head_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_channels * grid_size * grid_size, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, grid_size * grid_size * output_width),
        )
        self.output_width = output_width

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("expected a batch of RGB images with shape (N, 3, 224, 224)")
        if images.shape[-2:] != (224, 224):
            raise ValueError("YOLOv1 expects 224×224 input images")
        features = self.backbone(images)
        if features.ndim != 4:
            raise ValueError("backbone must return an NCHW feature map")
        logits = self.head(self.features(features))
        return torch.sigmoid(
            logits.reshape(-1, self.grid_size, self.grid_size, self.output_width)
        )
