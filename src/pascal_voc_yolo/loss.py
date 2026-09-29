"""YOLOv1-style masks and decomposed training objective."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .geometry import intersection_over_union


@dataclass(frozen=True)
class MaskBundle:
    """Per-cell and per-box responsibility masks plus target IoUs."""

    ious: torch.Tensor
    cell: torch.Tensor
    box: torch.Tensor


@dataclass(frozen=True)
class LossBundle:
    """Total YOLO loss and its three reported components."""

    total: torch.Tensor
    classification: torch.Tensor
    localization: torch.Tensor
    confidence: torch.Tensor


def _validate_tensors(
    predictions: torch.Tensor, targets: torch.Tensor, boxes_per_cell: int
) -> int:
    if predictions.shape != targets.shape or predictions.ndim != 4:
        raise ValueError("predictions and targets must share shape (N, S, S, 5B + C)")
    if boxes_per_cell <= 0:
        raise ValueError("boxes_per_cell must be positive")
    num_classes = predictions.shape[-1] - 5 * boxes_per_cell
    if num_classes <= 0:
        raise ValueError("tensor width must include at least one class after 5B box values")
    return num_classes


def _boxes(values: torch.Tensor, boxes_per_cell: int) -> torch.Tensor:
    return values[..., : 5 * boxes_per_cell].reshape(*values.shape[:-1], boxes_per_cell, 5)


def create_masks(
    predictions: torch.Tensor, targets: torch.Tensor, boxes_per_cell: int = 2
) -> MaskBundle:
    """Choose the highest-IoU predictor for each occupied grid cell."""

    _validate_tensors(predictions, targets, boxes_per_cell)
    predicted_boxes = _boxes(predictions, boxes_per_cell)
    target_boxes = _boxes(targets, boxes_per_cell)
    pairwise_ious = intersection_over_union(
        predicted_boxes[..., :, None, 1:5], target_boxes[..., None, :, 1:5]
    )
    ious = pairwise_ious.max(dim=-1).values
    cell_mask = target_boxes[..., 0, 0] > 0
    responsible_index = ious.argmax(dim=-1)
    responsible = F.one_hot(responsible_index, num_classes=boxes_per_cell).to(torch.bool)
    box_mask = responsible & cell_mask.unsqueeze(-1)
    return MaskBundle(ious=ious, cell=cell_mask, box=box_mask)


def classification_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    cell_mask: torch.Tensor,
    boxes_per_cell: int = 2,
) -> torch.Tensor:
    """Squared class-probability error for occupied cells."""

    _validate_tensors(predictions, targets, boxes_per_cell)
    offset = 5 * boxes_per_cell
    per_cell = (predictions[..., offset:] - targets[..., offset:]).square().sum(dim=-1)
    return (per_cell * cell_mask).sum()


def localization_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    box_mask: torch.Tensor,
    boxes_per_cell: int = 2,
) -> torch.Tensor:
    """Squared center and square-root size error for responsible boxes."""

    _validate_tensors(predictions, targets, boxes_per_cell)
    predicted_boxes = _boxes(predictions, boxes_per_cell)
    target_boxes = _boxes(targets, boxes_per_cell)
    center_error = (predicted_boxes[..., 1:3] - target_boxes[..., 1:3]).square().sum(dim=-1)
    size_error = (
        predicted_boxes[..., 3:5].clamp_min(0).sqrt()
        - target_boxes[..., 3:5].clamp_min(0).sqrt()
    ).square().sum(dim=-1)
    return ((center_error + size_error) * box_mask).sum()


def confidence_loss(
    predictions: torch.Tensor,
    ious: torch.Tensor,
    box_mask: torch.Tensor,
    boxes_per_cell: int = 2,
    lambda_no_object: float = 0.5,
) -> torch.Tensor:
    """Object and no-object confidence error."""

    predicted_confidence = _boxes(predictions, boxes_per_cell)[..., 0]
    object_error = (predicted_confidence - ious).square() * box_mask
    no_object_error = predicted_confidence.square() * ~box_mask
    return object_error.sum() + lambda_no_object * no_object_error.sum()


def yolo_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    boxes_per_cell: int = 2,
    lambda_coord: float = 5.0,
    lambda_no_object: float = 0.5,
) -> LossBundle:
    """Compute the YOLOv1-style objective and its reported components."""

    masks = create_masks(predictions, targets, boxes_per_cell)
    class_value = classification_loss(predictions, targets, masks.cell, boxes_per_cell)
    localization_value = lambda_coord * localization_loss(
        predictions, targets, masks.box, boxes_per_cell
    )
    confidence_value = confidence_loss(
        predictions,
        masks.ious,
        masks.box,
        boxes_per_cell,
        lambda_no_object,
    )
    return LossBundle(
        total=class_value + localization_value + confidence_value,
        classification=class_value,
        localization=localization_value,
        confidence=confidence_value,
    )
