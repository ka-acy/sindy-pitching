#!/usr/bin/env python3
"""Build a Stage B clip manifest from the MLB-YouTube metadata JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SOURCE_NAME = "mlb_youtube"
DEFAULT_VIEW_TYPE = "broadcast_centerfield"
DEFAULT_FPS = ""
DEFAULT_NUM_FRAMES = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def task_type_from_labels(labels: list[str]) -> str:
    labels_lower = {label.lower() for label in labels}
    if not labels_lower:
        return "broadcast_context"
    if {"swing", "hit", "foul", "in play", "bunt"} & labels_lower:
        return "batting"
    if {"ball", "strike", "hit by pitch"} & labels_lower:
        return "pitching"
    return "baseball_event"


def segmented_rows(data: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip_id, item in data.items():
        labels = item.get("labels", [])
        rows.append(
            {
                "clip_id": clip_id,
                "source_name": SOURCE_NAME,
                "task_type": task_type_from_labels(labels),
                "filepath": source_path.as_posix(),
                "modality": "video_metadata",
                "fps": DEFAULT_FPS,
                "num_frames": DEFAULT_NUM_FRAMES,
                "subject_id": "",
                "handedness": "",
                "view_type": DEFAULT_VIEW_TYPE,
                "event_start_frame": "",
                "event_end_frame": "",
                "notes": json.dumps(
                    {
                        "dataset_component": "segmented",
                        "subset": item.get("subset", ""),
                        "url": item.get("url", ""),
                        "video_start_sec": item.get("start", ""),
                        "video_end_sec": item.get("end", ""),
                        "labels": labels,
                        "pitch_type": item.get("type", ""),
                        "pitch_speed_mph": item.get("speed", ""),
                    },
                    sort_keys=True,
                ),
            }
        )
    return rows


def continuous_rows(data: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip_id, item in data.items():
        labels = sorted({annotation["label"] for annotation in item.get("annotations", [])})
        rows.append(
            {
                "clip_id": clip_id,
                "source_name": SOURCE_NAME,
                "task_type": task_type_from_labels(labels),
                "filepath": source_path.as_posix(),
                "modality": "video_metadata",
                "fps": DEFAULT_FPS,
                "num_frames": DEFAULT_NUM_FRAMES,
                "subject_id": "",
                "handedness": "",
                "view_type": DEFAULT_VIEW_TYPE,
                "event_start_frame": "",
                "event_end_frame": "",
                "notes": json.dumps(
                    {
                        "dataset_component": "continuous",
                        "subset": item.get("subset", ""),
                        "url": item.get("url", ""),
                        "video_start_sec": item.get("start", ""),
                        "video_end_sec": item.get("end", ""),
                        "labels": labels,
                        "annotations": item.get("annotations", []),
                    },
                    sort_keys=True,
                ),
            }
        )
    return rows


def captions_rows(data: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ytid, segments in data.items():
        rows.append(
            {
                "clip_id": f"captions_{ytid}",
                "source_name": SOURCE_NAME,
                "task_type": "broadcast_context",
                "filepath": source_path.as_posix(),
                "modality": "caption_metadata",
                "fps": DEFAULT_FPS,
                "num_frames": DEFAULT_NUM_FRAMES,
                "subject_id": "",
                "handedness": "",
                "view_type": "broadcast_audio",
                "event_start_frame": "",
                "event_end_frame": "",
                "notes": json.dumps(
                    {
                        "dataset_component": "captions",
                        "youtube_id": ytid,
                        "num_caption_segments": len(segments),
                    },
                    sort_keys=True,
                ),
            }
        )
    return rows


def negative_rows(data: dict[str, Any], source_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip_id, item in data.items():
        rows.append(
            {
                "clip_id": clip_id,
                "source_name": SOURCE_NAME,
                "task_type": "negative",
                "filepath": source_path.as_posix(),
                "modality": "video_metadata",
                "fps": DEFAULT_FPS,
                "num_frames": DEFAULT_NUM_FRAMES,
                "subject_id": "",
                "handedness": "",
                "view_type": DEFAULT_VIEW_TYPE,
                "event_start_frame": "",
                "event_end_frame": "",
                "notes": json.dumps(
                    {
                        "dataset_component": "negative",
                        "youtube_id": item.get("ytid", ""),
                        "interval_sec": item.get("interval", []),
                    },
                    sort_keys=True,
                ),
            }
        )
    return rows


def smoke_subset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["task_type"], []).append(row)

    selected: list[dict[str, Any]] = []
    for task_type in ("pitching", "batting", "negative", "broadcast_context"):
        selected.extend(grouped.get(task_type, [])[:3])
    return selected


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    metadata_root = project_root / "data" / "baseball" / "raw" / "mlb-youtube-master" / "data"

    segmented_path = metadata_root / "mlb-youtube-segmented.json"
    continuous_path = metadata_root / "mlb-youtube-continuous.json"
    captions_path = metadata_root / "mlb-youtube-captions.json"
    negative_path = metadata_root / "mlb-youtube-negative.json"

    segmented = json.load(segmented_path.open("r", encoding="utf-8"))
    continuous = json.load(continuous_path.open("r", encoding="utf-8"))
    captions = json.load(captions_path.open("r", encoding="utf-8"))
    negative = json.load(negative_path.open("r", encoding="utf-8"))

    rows = []
    rows.extend(segmented_rows(segmented, segmented_path.relative_to(project_root)))
    rows.extend(continuous_rows(continuous, continuous_path.relative_to(project_root)))
    rows.extend(captions_rows(captions, captions_path.relative_to(project_root)))
    rows.extend(negative_rows(negative, negative_path.relative_to(project_root)))
    rows.sort(key=lambda row: (row["task_type"], row["clip_id"]))

    metadata_out = project_root / "data" / "baseball" / "metadata" / "clip_manifest.csv"
    splits_root = project_root / "data" / "baseball" / "splits"

    fieldnames = [
        "clip_id",
        "source_name",
        "task_type",
        "filepath",
        "modality",
        "fps",
        "num_frames",
        "subject_id",
        "handedness",
        "view_type",
        "event_start_frame",
        "event_end_frame",
        "notes",
    ]
    write_csv(metadata_out, fieldnames, rows)
    write_csv(
        splits_root / "smoke_test.csv",
        ["clip_id", "source_name", "task_type", "filepath"],
        [
            {
                "clip_id": row["clip_id"],
                "source_name": row["source_name"],
                "task_type": row["task_type"],
                "filepath": row["filepath"],
            }
            for row in smoke_subset_rows(rows)
        ],
    )

    print(f"manifest rows: {len(rows)}")
    print(f"smoke rows: {len(smoke_subset_rows(rows))}")


if __name__ == "__main__":
    main()
