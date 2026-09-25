#!/usr/bin/env python3
"""Export a handedness-annotation template for Stage B baseball clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/baseball/splits/pitching_local_pool.csv"),
        help="Clip list CSV relative to project root.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/baseball/metadata/handedness_overrides.csv"),
        help="Output template CSV relative to project root.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    input_rows = read_csv(project_root / args.input_csv)
    rows = [
        {
            "clip_id": row["clip_id"],
            "task_type": row.get("task_type", ""),
            "video_url": row.get("video_url", ""),
            "pitch_type": row.get("pitch_type", ""),
            "pitch_speed_mph": row.get("pitch_speed_mph", ""),
            "handedness": "",
            "notes": "",
        }
        for row in input_rows
    ]
    write_csv(
        project_root / args.output_csv,
        ["clip_id", "task_type", "video_url", "pitch_type", "pitch_speed_mph", "handedness", "notes"],
        rows,
    )
    print(f"template rows: {len(rows)}")
    print(f"output: {(project_root / args.output_csv).relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
