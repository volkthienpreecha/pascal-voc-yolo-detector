import pytest
import torch

from pascal_voc_yolo.data import encode_voc_target


def test_box_is_encoded_in_expected_cell_and_class_slot() -> None:
    target = encode_voc_target(
        boxes=torch.tensor([[0.0, 0.0, 32.0, 32.0]]),
        labels=torch.tensor([2]),
        image_width=224,
        image_height=224,
        grid_size=7,
        boxes_per_cell=2,
        num_classes=20,
    )
    assert target.shape == (7, 7, 30)
    assert target[0, 0, 0].item() == 1
    assert target[0, 0, 5].item() == 1
    assert target[0, 0, 12].item() == 1
    assert target[..., 0].sum().item() == 1


def test_encoder_rejects_mismatched_boxes_and_labels() -> None:
    with pytest.raises(ValueError, match="same number"):
        encode_voc_target(
            torch.zeros(2, 4),
            torch.zeros(1, dtype=torch.long),
            224,
            224,
        )
