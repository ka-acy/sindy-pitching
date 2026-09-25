#!/usr/bin/env python3
"""Export concrete Stage B smoke-target metadata with URLs and time windows."""

from __future__ import annotations

import argparse
import csv
import json
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


def recommendation(task_type: str, notes: dict[str, Any]) -> str:
    component = notes.get("dataset_component", "")
    if task_type == "pitching" and component == "segmented":
        return "recommended_first_pose_target"
    if task_type == "batting" and component == "segmented":
        return "recommended_after_pitching"
    if task_type == "batting" and component == "continuous":
        return "longer_followup_target"
    return "context_or_negative_reference"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = {row["clip_id"]: row for row in read_csv(root / "data" / "baseball" / "metadata" / "clip_manifest.csv")}
    smoke_rows = read_csv(root / "data" / "baseball" / "splits" / "smoke_test.csv")

    out_rows: list[dict[str, Any]] = []
    for row in smoke_rows:
        full = manifest[row["clip_id"]]
        notes = json.loads(full["notes"])
        out_rows.append(
            {
                "clip_id": full["clip_id"],
                "task_type": full["task_type"],
                "dataset_component": notes.get("dataset_component", ""),
                "url": notes.get("url", ""),
                "video_start_sec": notes.get("video_start_sec", ""),
                "video_end_sec": notes.get("video_end_sec", ""),
                "labels": "|".join(notes.get("labels", [])),
                "pitch_type": notes.get("pitch_type", ""),
                "pitch_speed_mph": notes.get("pitch_speed_mph", ""),
                "subset": notes.get("subset", ""),
                "recommendation": recommendation(full["task_type"], notes),
            }
        )

    write_csv(
        root / "data" / "baseball" / "splits" / "smoke_targets_detailed.csv",
        [
            "clip_id",
            "task_type",
            "dataset_component",
            "url",
            "video_start_sec",
            "video_end_sec",
            "labels",
            "pitch_type",
            "pitch_speed_mph",
            "subset",
            "recommendation",
        ],
        out_rows,
    )
    print(f"detailed smoke targets: {len(out_rows)}")


if __name__ == "__main__":
    main()
