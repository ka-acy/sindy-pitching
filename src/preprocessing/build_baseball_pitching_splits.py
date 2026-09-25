#!/usr/bin/env python3
"""Build practical local pitching train/test splits for Stage B baseball modeling."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=30,
        help="Total number of local pitching clips to include in the initial pool.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=6,
        help="Number of clips to reserve for test.",
    )
    parser.add_argument(
        "--skip-pose-filter",
        action="store_true",
        help=(
            "Do not require clips to already have a usable standardized-pose "
            "payload. Use this when growing the pool before running pose "
            "extraction on the new candidates."
        ),
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


def read_usable_pose_clip_ids(project_root: Path) -> set[str]:
    manifest_path = project_root / "data" / "baseball" / "standardized_pose" / "manifest.csv"
    if not manifest_path.exists():
        return set()

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    usable_ids: set[str] = set()
    for row in rows:
        payload = np.load(project_root / row["filepath"])
        metadata = json.loads(str(payload["metadata"]))
        quality_flags = {str(flag) for flag in metadata.get("quality_flags", [])}
        if "template_payload" in quality_flags:
            continue
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        finite = coordinates[np.isfinite(coordinates)]
        if finite.size == 0 or float(np.nanstd(finite)) < 1e-8:
            continue
        usable_ids.add(row["clip_id"])
    return usable_ids


def score_row(notes: dict[str, Any]) -> tuple[float, float]:
    speed = float(notes.get("pitch_speed_mph") or 0.0)
    start = float(notes.get("video_start_sec") or 0.0)
    return (-speed, start)


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    usable_pose_clip_ids: set[str] = set() if args.skip_pose_filter else read_usable_pose_clip_ids(project_root)

    local_manifest = {
        row["clip_id"]: row
        for row in read_csv(project_root / "data" / "baseball" / "metadata" / "local_media_manifest.csv")
        if row["local_media_exists"] == "True"
    }
    clip_rows = read_csv(project_root / "data" / "baseball" / "metadata" / "clip_manifest.csv")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clip_rows:
        if row["task_type"] != "pitching" or row["clip_id"] not in local_manifest:
            continue
        if usable_pose_clip_ids and row["clip_id"] not in usable_pose_clip_ids:
            continue
        notes = json.loads(row["notes"])
        if notes.get("dataset_component") != "segmented":
            continue
        start_sec = float(notes.get("video_start_sec") or 0.0)
        end_sec = float(notes.get("video_end_sec") or 0.0)
        pitch_type = (notes.get("pitch_type") or "").strip()
        pitch_speed = notes.get("pitch_speed_mph")
        if start_sec < 0.0 or end_sec <= start_sec:
            continue
        if not pitch_type or pitch_speed in ("", None):
            continue
        pitch_type = (notes.get("pitch_type") or "unknown").lower()
        grouped[pitch_type].append(
            {
                "clip_id": row["clip_id"],
                "source_name": row["source_name"],
                "task_type": row["task_type"],
                "local_media_path": local_manifest[row["clip_id"]]["local_media_path"],
                "local_media_exists": True,
                "pitch_type": notes.get("pitch_type", ""),
                "pitch_speed_mph": notes.get("pitch_speed_mph", ""),
                "video_url": notes.get("url", ""),
                "video_start_sec": start_sec,
                "video_end_sec": end_sec,
                "view_type": row["view_type"],
            }
        )

    for rows in grouped.values():
        rows.sort(key=lambda row: score_row({"pitch_speed_mph": row["pitch_speed_mph"], "video_start_sec": row["video_start_sec"]}))

    ordered_types = sorted(grouped, key=lambda name: (-len(grouped[name]), name))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    index = 0
    while len(selected) < args.pool_size and ordered_types:
        pitch_type = ordered_types[index % len(ordered_types)]
        rows = grouped[pitch_type]
        if rows:
            row = rows.pop(0)
            if row["clip_id"] not in seen:
                selected.append(row)
                seen.add(row["clip_id"])
        index += 1
        ordered_types = [name for name in ordered_types if grouped[name]]

    selected.sort(key=lambda row: ((row["pitch_type"] or "").lower(), -(float(row["pitch_speed_mph"] or 0.0)), row["clip_id"]))

    # Stratify the test split across pitch types so every pitch type that
    # appears in the pool also appears in train AND test. Without this, an
    # alphabetical head-of-pool slice can fill the test set with one or two
    # pitch types the train set then never sees.
    by_pitch_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_pitch_type[(row["pitch_type"] or "").lower()].append(row)

    pool_size = len(selected) or 1
    test_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for pitch_type, rows in by_pitch_type.items():
        # Proportional allocation, with a floor of 1 test clip per pitch type
        # whenever the pool has more than one clip of that type.
        target_test = max(1, round(len(rows) * args.test_size / pool_size)) if len(rows) > 1 else 0
        test_rows.extend(rows[:target_test])
        train_rows.extend(rows[target_test:])

    # If proportional rounding overshot test_size, demote excess test clips
    # back to train (largest pitch-type buckets first so we keep coverage).
    if len(test_rows) > args.test_size:
        test_rows.sort(key=lambda row: ((row["pitch_type"] or "").lower(), -(float(row["pitch_speed_mph"] or 0.0)), row["clip_id"]))
        excess = len(test_rows) - args.test_size
        train_rows.extend(test_rows[-excess:])
        test_rows = test_rows[:-excess]

    test_rows.sort(key=lambda row: ((row["pitch_type"] or "").lower(), -(float(row["pitch_speed_mph"] or 0.0)), row["clip_id"]))
    train_rows.sort(key=lambda row: ((row["pitch_type"] or "").lower(), -(float(row["pitch_speed_mph"] or 0.0)), row["clip_id"]))

    fieldnames = [
        "clip_id",
        "source_name",
        "task_type",
        "local_media_path",
        "local_media_exists",
        "pitch_type",
        "pitch_speed_mph",
        "video_url",
        "video_start_sec",
        "video_end_sec",
        "view_type",
    ]
    splits_root = project_root / "data" / "baseball" / "splits"
    write_csv(splits_root / "pitching_local_pool.csv", fieldnames, selected)
    write_csv(splits_root / "pitching_local_train.csv", fieldnames, train_rows)
    write_csv(splits_root / "pitching_local_test.csv", fieldnames, test_rows)

    print(f"selected pool: {len(selected)}")
    print(f"train clips: {len(train_rows)}")
    print(f"test clips: {len(test_rows)}")


if __name__ == "__main__":
    main()
