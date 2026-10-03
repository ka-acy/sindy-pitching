#!/usr/bin/env python3
"""Create a Stage B standardized pose template payload."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


JOINT_NAMES = [
    "head",
    "neck",
    "spine",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root.",
    )
    parser.add_argument(
        "--clip-id",
        default="example_clip",
        help="Clip id to scaffold.",
    )
    parser.add_argument(
        "--source-name",
        default="",
        help="Source name override. If blank, use pose_jobs.csv when available.",
    )
    parser.add_argument(
        "--pose-jobs-csv",
        type=Path,
        default=Path("data/baseball/metadata/pose_jobs.csv"),
        help="Pose-job CSV to use for clip-specific output routing.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=0,
        help="Frame count override. If omitted, infer from pose_jobs.csv or use 5.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS to store in metadata when not specified elsewhere.",
    )
    parser.add_argument(
        "--pose-source",
        default="template_pending_coordinates",
        help="Pose source label stored in metadata.",
    )
    parser.add_argument(
        "--coordinate-dims",
        type=int,
        default=2,
        choices=(2, 3),
        help="Coordinate dimensionality for the scaffold.",
    )
    return parser.parse_args()


def read_pose_jobs(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["clip_id"]: row for row in rows}


def infer_frame_count(job: dict[str, str], fps: float) -> int:
    start_value = job.get("video_start_sec", "")
    end_value = job.get("video_end_sec", "")
    if not start_value or not end_value:
        return 5
    duration = max(0.0, float(end_value) - float(start_value))
    if duration <= 0.0:
        return 5
    return max(1, int(math.ceil(duration * fps)))


def build_metadata(
    args: argparse.Namespace,
    job: dict[str, str] | None,
    source_name: str,
    frame_count: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "clip_id": args.clip_id,
        "source_name": source_name,
        "task_type": "batting",
        "fps": args.fps,
        "dt": 1.0 / args.fps,
        "coordinate_dims": args.coordinate_dims,
        "joint_schema": "preferred_15_joint",
        "pose_source": args.pose_source,
        "subject_id": "",
        "handedness": "",
        "view_type": "broadcast_centerfield",
        "confidence_available": False,
        "tracking_notes": (
            "Template payload for Stage B adapter development. "
            "Replace coordinates and confirm frame count before registration."
        ),
        "quality_flags": ["template_payload", "replace_before_registration"],
        "template_frame_count": frame_count,
    }
    if not job:
        return metadata

    metadata.update(
        {
            "task_type": job.get("task_type", metadata["task_type"]),
            "source_name": job.get("source_name", source_name),
            "coordinate_dims": int(job.get("coordinate_dims", args.coordinate_dims)),
            "pose_source": job.get("pose_source", args.pose_source),
            "subject_id": job.get("subject_id", ""),
            "handedness": job.get("handedness", ""),
            "view_type": job.get("view_type", metadata["view_type"]),
            "local_media_path": job.get("local_media_path", ""),
            "video_url": job.get("video_url", ""),
            "video_start_sec": job.get("video_start_sec", ""),
            "video_end_sec": job.get("video_end_sec", ""),
            "event_start_frame": job.get("event_start_frame", ""),
            "event_end_frame": job.get("event_end_frame", ""),
            "pitch_type": job.get("pitch_type", ""),
            "pitch_speed_mph": job.get("pitch_speed_mph", ""),
            "labels": json.loads(job.get("labels", "[]")),
            "tracking_notes": (
                "Template payload built from pose_jobs.csv. "
                "Replace coordinates and verify fps/frame count before registration."
            ),
        }
    )
    return metadata


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    pose_jobs = read_pose_jobs(project_root / args.pose_jobs_csv)
    job = pose_jobs.get(args.clip_id)
    source_name = args.source_name or (job["source_name"] if job else "example_source")
    output_path = (
        project_root / job["pose_output_path"]
        if job and job.get("pose_output_path")
        else project_root / "data" / "baseball" / "standardized_pose" / source_name / f"{args.clip_id}.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = args.frame_count or infer_frame_count(job, args.fps) if job else args.frame_count or 5
    joint_count = len(JOINT_NAMES)
    coordinate_dims = int(job.get("coordinate_dims", args.coordinate_dims)) if job else args.coordinate_dims
    coordinates = np.zeros((frame_count, joint_count, coordinate_dims), dtype=np.float32)
    metadata = build_metadata(args, job, source_name, frame_count)

    np.savez_compressed(
        output_path,
        coordinates=coordinates,
        joint_names=np.array(JOINT_NAMES, dtype="<U32"),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(output_path.relative_to(project_root).as_posix())


if __name__ == "__main__":
    main()
