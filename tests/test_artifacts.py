from dataclasses import asdict

import pytest

from pascal_voc_yolo.artifacts import normalize_prediction_rows


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "data_num": " 3 ",
        "class_label": " person ",
        "class_num": " 14 ",
        "x1": " 10.5 ",
        "y1": " 20.25 ",
        "x2": " 100.0 ",
        "y2": " 150.75 ",
    }
    row.update(overrides)
    return row


def test_normalizer_trims_and_converts_to_canonical_fields() -> None:
    normalized = normalize_prediction_rows([_row()])
    assert asdict(normalized[0]) == {
        "data_num": 3,
        "class_label": "person",
        "class_num": 14,
        "x1": 10.5,
        "y1": 20.25,
        "x2": 100.0,
        "y2": 150.75,
    }


def test_normalizer_preserves_row_order() -> None:
    normalized = normalize_prediction_rows([_row(data_num="2"), _row(data_num="1")])
    assert [row.data_num for row in normalized] == [2, 1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"class_num": "20"},
        {"x1": "nan"},
        {"x1": "101", "x2": "100"},
        {"y1": "151", "y2": "150"},
        {"data_num": "not-an-integer"},
    ],
)
def test_normalizer_rejects_invalid_rows(overrides: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        normalize_prediction_rows([_row(**overrides)])


def test_normalizer_rejects_missing_column() -> None:
    row = _row()
    del row["x2"]
    with pytest.raises(ValueError, match="columns"):
        normalize_prediction_rows([row])
