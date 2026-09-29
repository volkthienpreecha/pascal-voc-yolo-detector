import pytest
import torch

from pascal_voc_yolo.geometry import intersection_over_union


def test_identical_boxes_have_unit_iou() -> None:
    box = torch.tensor([0.5, 0.5, 0.4, 0.2])
    assert intersection_over_union(box, box).item() == pytest.approx(1.0)


def test_disjoint_boxes_have_zero_iou() -> None:
    first = torch.tensor([0.1, 0.1, 0.1, 0.1])
    second = torch.tensor([0.9, 0.9, 0.1, 0.1])
    assert intersection_over_union(first, second).item() == pytest.approx(0.0)


def test_zero_area_boxes_return_finite_zero() -> None:
    zero = torch.tensor([0.5, 0.5, 0.0, 0.0])
    result = intersection_over_union(zero, zero)
    assert result.item() == pytest.approx(0.0)
    assert torch.isfinite(result)


def test_iou_rejects_invalid_trailing_dimension() -> None:
    with pytest.raises(ValueError, match="four"):
        intersection_over_union(torch.zeros(3), torch.zeros(3))
