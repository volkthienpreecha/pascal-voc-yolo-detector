import torch

from pascal_voc_yolo.loss import create_masks, yolo_loss


def _occupied_cell() -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.zeros(1, 1, 1, 13)
    targets = torch.zeros_like(predictions)
    targets[..., 0:5] = torch.tensor([1.0, 0.5, 0.5, 0.4, 0.4])
    targets[..., 5:10] = torch.tensor([1.0, 0.5, 0.5, 0.4, 0.4])
    targets[..., 10] = 1.0
    predictions[..., 0:5] = torch.tensor([0.8, 0.5, 0.5, 0.4, 0.4])
    predictions[..., 5:10] = torch.tensor([0.2, 0.1, 0.1, 0.1, 0.1])
    predictions[..., 10:] = torch.tensor([0.9, 0.05, 0.05])
    return predictions, targets


def test_masks_select_exactly_one_responsible_box() -> None:
    predictions, targets = _occupied_cell()
    masks = create_masks(predictions, targets, boxes_per_cell=2)
    assert masks.cell.shape == (1, 1, 1)
    assert masks.box.shape == (1, 1, 1, 2)
    assert masks.box.sum().item() == 1
    assert masks.box[..., 0].item()


def test_empty_targets_have_zero_class_and_localization_loss() -> None:
    predictions = torch.zeros(2, 2, 2, 13)
    targets = torch.zeros_like(predictions)
    losses = yolo_loss(predictions, targets, boxes_per_cell=2)
    assert losses.classification.item() == 0
    assert losses.localization.item() == 0


def test_every_loss_component_is_a_finite_scalar() -> None:
    predictions, targets = _occupied_cell()
    losses = yolo_loss(predictions, targets, boxes_per_cell=2)
    for value in (
        losses.total,
        losses.classification,
        losses.localization,
        losses.confidence,
    ):
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_masks_reject_invalid_prediction_width() -> None:
    predictions = torch.zeros(1, 1, 1, 10)
    targets = torch.zeros_like(predictions)
    try:
        create_masks(predictions, targets, boxes_per_cell=2)
    except ValueError as error:
        assert "class" in str(error)
    else:
        raise AssertionError("invalid tensor width was accepted")
