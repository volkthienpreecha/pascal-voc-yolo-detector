import pytest

from pascal_voc_yolo.postprocess import Detection, non_max_suppression


def test_nms_removes_lower_scored_overlap_for_same_class() -> None:
    detections = [
        Detection(0, 0, 10, 10, 0.9, 3, 0),
        Detection(1, 1, 9, 9, 0.7, 3, 1),
    ]
    assert non_max_suppression(detections, 0.5, 0.1) == [detections[0]]


def test_nms_keeps_overlapping_boxes_from_different_classes() -> None:
    detections = [
        Detection(0, 0, 10, 10, 0.9, 3, 0),
        Detection(1, 1, 9, 9, 0.7, 4, 1),
    ]
    assert non_max_suppression(detections, 0.5, 0.1) == detections


def test_nms_uses_source_index_as_stable_tiebreaker() -> None:
    later = Detection(20, 20, 30, 30, 0.8, 1, 5)
    earlier = Detection(0, 0, 10, 10, 0.8, 1, 2)
    assert non_max_suppression([later, earlier], 0.5, 0.1) == [earlier, later]


@pytest.mark.parametrize(
    ("iou_threshold", "score_threshold"),
    [(-0.1, 0.2), (1.1, 0.2), (0.5, -0.1), (0.5, 1.1)],
)
def test_nms_rejects_invalid_thresholds(iou_threshold: float, score_threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        non_max_suppression([], iou_threshold, score_threshold)
