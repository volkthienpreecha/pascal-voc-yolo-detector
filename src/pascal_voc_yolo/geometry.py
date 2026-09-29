"""Bounding-box geometry for YOLO-style center-width-height boxes."""

import torch


def intersection_over_union(
    boxes1: torch.Tensor, boxes2: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """Compute broadcastable IoU for ``(..., x, y, width, height)`` boxes."""

    if boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError("boxes must have four trailing coordinates: x, y, width, height")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    center1, size1 = boxes1[..., :2], boxes1[..., 2:].clamp_min(0)
    center2, size2 = boxes2[..., :2], boxes2[..., 2:].clamp_min(0)
    top_left = torch.maximum(center1 - size1 / 2, center2 - size2 / 2)
    bottom_right = torch.minimum(center1 + size1 / 2, center2 + size2 / 2)
    intersection_size = (bottom_right - top_left).clamp_min(0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area1 = size1[..., 0] * size1[..., 1]
    area2 = size2[..., 0] * size2[..., 1]
    union = area1 + area2 - intersection
    return torch.where(union > 0, intersection / union.clamp_min(epsilon), torch.zeros_like(union))
