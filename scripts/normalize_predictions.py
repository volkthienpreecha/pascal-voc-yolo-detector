"""Normalize the detector's raw prediction CSV into a portable artifact."""

import argparse
import csv
from pathlib import Path

from pascal_voc_yolo.artifacts import normalize_prediction_rows, write_prediction_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with args.source.open(newline="", encoding="utf-8") as handle:
        rows = normalize_prediction_rows(csv.DictReader(handle))
    write_prediction_csv(rows, args.destination)
    print(f"wrote {len(rows):,} rows to {args.destination}")


if __name__ == "__main__":
    main()
