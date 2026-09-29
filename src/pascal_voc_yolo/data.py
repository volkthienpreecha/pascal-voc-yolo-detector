"""Pascal VOC annotation conversion for the YOLO grid representation."""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def encode_voc_target(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    image_width: int,
    image_height: int,
    grid_size: int = 7,
    boxes_per_cell: int = 2,
    num_classes: int = 20,
) -> torch.Tensor:
    """Encode absolute ``xyxy`` boxes and class IDs into a YOLO target grid."""

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if labels.ndim != 1 or boxes.shape[0] != labels.shape[0]:
        raise ValueError("boxes and labels must contain the same number of objects")
    if min(image_width, image_height, grid_size, boxes_per_cell, num_classes) <= 0:
        raise ValueError("image and grid dimensions must be positive")
    target = torch.zeros(grid_size, grid_size, 5 * boxes_per_cell + num_classes)
    for box, label_value in zip(boxes, labels, strict=True):
        class_id = int(label_value.item())
        if not 0 <= class_id < num_classes:
            raise ValueError("class label is outside the configured range")
        x1, y1, x2, y2 = (float(value) for value in box)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box coordinates must satisfy x1 < x2 and y1 < y2")
        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        column = min(grid_size - 1, max(0, int(center_x * grid_size)))
        row = min(grid_size - 1, max(0, int(center_y * grid_size)))
        box_values = torch.tensor(
            [1.0, center_x * grid_size - column, center_y * grid_size - row, width, height]
        )
        for box_index in range(boxes_per_cell):
            start = box_index * 5
            target[row, column, start : start + 5] = box_values
        target[row, column, 5 * boxes_per_cell + class_id] = 1.0
    return target


def parse_voc_annotation(annotation: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Extract boxes, class IDs, and source dimensions from torchvision VOC metadata."""

    root = annotation["annotation"]
    size = root["size"]
    objects = root.get("object", [])
    if isinstance(objects, dict):
        objects = [objects]
    boxes = []
    labels = []
    for item in objects:
        bbox = item["bndbox"]
        boxes.append([float(bbox[key]) for key in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(VOC_CLASSES.index(item["name"]))
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.long),
        int(size["width"]),
        int(size["height"]),
    )


class VOCGridDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Adapt a torchvision VOCDetection dataset to image/grid pairs."""

    def __init__(self, dataset: Dataset[Any], image_transform: Callable[[Any], torch.Tensor]):
        self.dataset = dataset
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, annotation = self.dataset[index]
        boxes, labels, width, height = parse_voc_annotation(annotation)
        return self.image_transform(image), encode_voc_target(boxes, labels, width, height)
