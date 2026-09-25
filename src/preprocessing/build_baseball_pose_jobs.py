#!/usr/bin/env python3
"""Build Stage B pose-extraction job manifests from local baseball media."""

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
    parser.add_argument(
        "--input-manifest",
        default="data/baseball/splits/smoke_test_local.csv",
        help="CSV of local media targets relative to project root.",
    )
    parser.add_argument(
        "--output-path",
        default="data/baseball/metadata/pose_jobs.csv",
        help="Pose job CSV path relative to project root.",
    )
    parser.add_argument(
        "--coordinate-dims",
        type=int,
        default=2,
        choices=(2, 3),
        help="Expected pose output dimensionality.",
    )
    parser.add_argument(
        "--pose-source",
        default="manual_pending",
        help="Pose extractor identifier placeholder.",
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
    local_targets = read_csv(project_root / args.input_manifest)
    clip_manifest = {
        row["clip_id"]: row
        for row in read_csv(project_root / "data" / "baseball" / "metadata" / "clip_manifest.csv")
    }

    rows: list[dict[str, Any]] = []
    for local_row in local_targets:
        local_media_exists = local_row.get("local_media_exists")
        if local_media_exists is None:
            local_media_exists = "True" if local_row.get("local_media_path") else "False"
        if local_media_exists != "True":
            continue
        clip_id = local_row["clip_id"]
        clip_row = clip_manifest[clip_id]
        notes = json.loads(clip_row["notes"])
        output_path = (
            project_root
            / "data"
            / "baseball"
            / "standardized_pose"
            / clip_row["source_name"]
            / f"{clip_id}.npz"
        )
        rows.append(
            {
                "clip_id": clip_id,
                "source_name": clip_row["source_name"],
                "task_type": clip_row["task_type"],
                "local_media_path": local_row["local_media_path"],
                "pose_output_path": output_path.relative_to(project_root).as_posix(),
                "pose_source": args.pose_source,
                "coordinate_dims": args.coordinate_dims,
                "view_type": clip_row["view_type"],
                "subject_id": clip_row["subject_id"],
                "handedness": clip_row["handedness"],
                "event_start_frame": clip_row["event_start_frame"],
                "event_end_frame": clip_row["event_end_frame"],
                "dataset_component": notes.get("dataset_component", ""),
                "video_url": notes.get("url", ""),
                "video_start_sec": notes.get("video_start_sec", ""),
                "video_end_sec": notes.get("video_end_sec", ""),
                "pitch_type": notes.get("pitch_type", ""),
                "pitch_speed_mph": notes.get("pitch_speed_mph", ""),
                "labels": json.dumps(notes.get("labels", []), sort_keys=True),
                "job_status": "pending_pose_extraction",
            }
        )

    write_csv(
        project_root / args.output_path,
        [
            "clip_id",
            "source_name",
            "task_type",
            "local_media_path",
            "pose_output_path",
            "pose_source",
            "coordinate_dims",
            "view_type",
            "subject_id",
            "handedness",
            "event_start_frame",
            "event_end_frame",
            "dataset_component",
            "video_url",
            "video_start_sec",
            "video_end_sec",
            "pitch_type",
            "pitch_speed_mph",
            "labels",
            "job_status",
        ],
        rows,
    )

    print(f"pose jobs: {len(rows)}")


if __name__ == "__main__":
    main()
