#!/usr/bin/env python3
"""Validate and register Stage B standardized pose payloads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_METADATA_KEYS = {
    "clip_id",
    "source_name",
    "task_type",
    "fps",
    "dt",
    "coordinate_dims",
    "joint_schema",
    "pose_source",
    "subject_id",
    "handedness",
    "view_type",
    "confidence_available",
}

PREFERRED_JOINTS = [
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
        "--input-glob",
        default="data/baseball/standardized_pose/*/*.npz",
        help="Glob for standardized pose payloads relative to project root.",
    )
    parser.add_argument(
        "--allow-unregistered",
        action="store_true",
        help="Allow payloads whose clip_id is not present in clip_manifest.csv.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_clip_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["clip_id"]: row for row in rows}


def validate_payload(
    path: Path,
    clip_manifest: dict[str, dict[str, str]],
    project_root: Path,
    allow_unregistered: bool,
) -> dict[str, Any]:
    payload = np.load(path)
    required_arrays = {"coordinates", "joint_names", "metadata"}
    missing = required_arrays.difference(payload.files)
    if missing:
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")

    coordinates = np.asarray(payload["coordinates"])
    joint_names = payload["joint_names"].tolist()
    metadata = json.loads(str(payload["metadata"]))

    missing_meta = REQUIRED_METADATA_KEYS.difference(metadata)
    if missing_meta:
        raise ValueError(f"{path}: missing metadata keys {sorted(missing_meta)}")

    if coordinates.ndim != 3:
        raise ValueError(f"{path}: expected coordinates with ndim=3, got {coordinates.shape}")
    if coordinates.shape[1] != len(joint_names):
        raise ValueError(f"{path}: joint dimension mismatch {coordinates.shape} vs {len(joint_names)}")
    if coordinates.shape[2] not in (2, 3):
        raise ValueError(f"{path}: unsupported coordinate dims {coordinates.shape[2]}")
    if int(metadata["coordinate_dims"]) != coordinates.shape[2]:
        raise ValueError(f"{path}: metadata coordinate_dims mismatch")

    if metadata["clip_id"] not in clip_manifest and not allow_unregistered:
        raise ValueError(f"{path}: clip_id {metadata['clip_id']} missing from clip manifest")

    if list(joint_names) == PREFERRED_JOINTS:
        joint_schema = "preferred_15_joint"
    else:
        joint_schema = "custom"

    return {
        "clip_id": metadata["clip_id"],
        "source_name": metadata["source_name"],
        "task_type": metadata["task_type"],
        "filepath": path.relative_to(project_root).as_posix(),
        "fps": metadata["fps"],
        "dt": metadata["dt"],
        "num_frames": int(coordinates.shape[0]),
        "num_joints": int(coordinates.shape[1]),
        "coordinate_dims": int(coordinates.shape[2]),
        "joint_schema_status": joint_schema,
        "registered_clip_id": metadata["clip_id"] in clip_manifest,
    }


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    clip_manifest = read_clip_manifest(project_root / "data" / "baseball" / "metadata" / "clip_manifest.csv")
    payload_paths = sorted(project_root.glob(args.input_glob))

    rows: list[dict[str, Any]] = []
    for path in payload_paths:
        rows.append(validate_payload(path, clip_manifest, project_root, args.allow_unregistered))

    write_csv(
        project_root / "data" / "baseball" / "standardized_pose" / "manifest.csv",
        [
            "clip_id",
            "source_name",
            "task_type",
            "filepath",
            "fps",
            "dt",
            "num_frames",
            "num_joints",
            "coordinate_dims",
            "joint_schema_status",
            "registered_clip_id",
        ],
        rows,
    )

    print(f"registered payloads: {len(rows)}")


if __name__ == "__main__":
    main()
