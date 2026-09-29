"""Prediction decoding and class-aware non-max suppression."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Detection:
    """One absolute-coordinate object detection."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    source_index: int


def _xyxy_iou(first: Detection, second: Detection) -> float:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def non_max_suppression(
    detections: Sequence[Detection], iou_threshold: float, score_threshold: float
) -> list[Detection]:
    """Suppress lower-scored overlaps within each class."""

    if not 0 <= iou_threshold <= 1 or not 0 <= score_threshold <= 1:
        raise ValueError("threshold values must be between zero and one")
    candidates = sorted(
        (item for item in detections if item.score >= score_threshold),
        key=lambda item: (-item.score, item.source_index),
    )
    selected: list[Detection] = []
    for candidate in candidates:
        if all(
            candidate.class_id != prior.class_id
            or _xyxy_iou(candidate, prior) <= iou_threshold
            for prior in selected
        ):
            selected.append(candidate)
    return selected


def decode_predictions(
    prediction: torch.Tensor,
    image_size: int = 224,
    boxes_per_cell: int = 2,
    score_threshold: float = 0.2,
) -> list[Detection]:
    """Decode one ``(S, S, 5B + C)`` grid into absolute-coordinate detections."""

    if prediction.ndim != 3:
        raise ValueError("prediction must have shape (S, S, 5B + C)")
    grid_size = prediction.shape[0]
    if prediction.shape[1] != grid_size:
        raise ValueError("prediction grid must be square")
    num_classes = prediction.shape[-1] - 5 * boxes_per_cell
    if num_classes <= 0:
        raise ValueError("prediction width does not contain class probabilities")
    class_probabilities = prediction[..., 5 * boxes_per_cell :]
    detections: list[Detection] = []
    source_index = 0
    for row in range(grid_size):
        for column in range(grid_size):
            class_probability, class_id = class_probabilities[row, column].max(dim=0)
            for box_index in range(boxes_per_cell):
                start = 5 * box_index
                confidence, x, y, width, height = prediction[row, column, start : start + 5]
                score = float((confidence * class_probability).item())
                if score >= score_threshold:
                    center_x = (column + float(x.item())) / grid_size * image_size
                    center_y = (row + float(y.item())) / grid_size * image_size
                    absolute_width = float(width.item()) * image_size
                    absolute_height = float(height.item()) * image_size
                    detections.append(
                        Detection(
                            center_x - absolute_width / 2,
                            center_y - absolute_height / 2,
                            center_x + absolute_width / 2,
                            center_y + absolute_height / 2,
                            score,
                            int(class_id.item()),
                            source_index,
                        )
                    )
                source_index += 1
    return detections
