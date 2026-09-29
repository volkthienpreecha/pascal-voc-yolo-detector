"""Validation and serialization for portable detector prediction artifacts."""

import csv
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictionRow:
    data_num: int
    class_label: str
    class_num: int
    x1: float
    y1: float
    x2: float
    y2: float


FIELDS = tuple(PredictionRow.__annotations__)


def normalize_prediction_rows(
    rows: Iterable[Mapping[str, str]],
) -> list[PredictionRow]:
    """Convert raw CSV mappings into validated, typed prediction rows."""

    normalized = []
    for row_number, row in enumerate(rows, start=2):
        if not set(FIELDS).issubset(row):
            raise ValueError(f"row {row_number} does not contain the required columns")
        try:
            item = PredictionRow(
                data_num=int(row["data_num"].strip()),
                class_label=row["class_label"].strip(),
                class_num=int(row["class_num"].strip()),
                x1=float(row["x1"].strip()),
                y1=float(row["y1"].strip()),
                x2=float(row["x2"].strip()),
                y2=float(row["y2"].strip()),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"row {row_number} contains an invalid value") from error
        if item.data_num < 0 or not 0 <= item.class_num < 20 or not item.class_label:
            raise ValueError(f"row {row_number} contains an invalid identifier")
        if not all(math.isfinite(value) for value in (item.x1, item.y1, item.x2, item.y2)):
            raise ValueError(f"row {row_number} contains a non-finite coordinate")
        if item.x1 > item.x2 or item.y1 > item.y2:
            raise ValueError(f"row {row_number} contains reversed coordinates")
        normalized.append(item)
    return normalized


def write_prediction_csv(rows: Iterable[PredictionRow], destination: Path) -> None:
    """Write predictions using a stable seven-column UTF-8 CSV schema."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
