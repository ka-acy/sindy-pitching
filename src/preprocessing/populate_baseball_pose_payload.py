#!/usr/bin/env python3
"""Populate a Stage B baseball pose payload from external coordinates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


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

MEDIAPIPE_JOINTS = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
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
        required=True,
        help="Target clip id.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="External pose file (.npy or .npz).",
    )
    parser.add_argument(
        "--coordinates-key",
        default="coordinates",
        help="Array key for coordinates when using .npz input.",
    )
    parser.add_argument(
        "--joint-names-key",
        default="joint_names",
        help="Array key for joint names when using .npz input.",
    )
    parser.add_argument(
        "--joint-names",
        default="",
        help="Comma-separated joint names override when input does not contain them.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override fps stored in metadata.",
    )
    parser.add_argument(
        "--pose-source",
        default="external_pose_import",
        help="Pose source label stored in metadata.",
    )
    parser.add_argument(
        "--pose-jobs-csv",
        type=Path,
        default=Path("data/baseball/metadata/pose_jobs.csv"),
        help="Pose-job CSV used to resolve the target payload path.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_coordinates(path: Path, coordinates_key: str, joint_names_key: str) -> tuple[np.ndarray, list[str] | None]:
    if path.suffix == ".npy":
        return np.load(path), None
    if path.suffix == ".npz":
        payload = np.load(path)
        coordinates = np.asarray(payload[coordinates_key])
        joint_names = payload[joint_names_key].tolist() if joint_names_key in payload.files else None
        return coordinates, joint_names
    raise ValueError(f"Unsupported input file type: {path.suffix}")


def load_input_metadata(path: Path) -> dict[str, Any]:
    if path.suffix != ".npz":
        return {}
    payload = np.load(path)
    if "metadata" not in payload.files:
        return {}
    return json.loads(str(payload["metadata"]))


def read_pose_job(project_root: Path, pose_jobs_csv: Path, clip_id: str) -> dict[str, str] | None:
    pose_jobs_path = project_root / pose_jobs_csv
    if not pose_jobs_path.exists():
        return None
    for row in read_csv(pose_jobs_path):
        if row["clip_id"] == clip_id:
            return row
    return None


def parse_joint_names(value: str) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def reorder_to_preferred(coordinates: np.ndarray, joint_names: list[str]) -> np.ndarray:
    if len(joint_names) != coordinates.shape[1]:
        raise ValueError("Joint-name count does not match coordinate joint dimension")
    index_by_name = {name: idx for idx, name in enumerate(joint_names)}
    missing = [name for name in PREFERRED_JOINTS if name not in index_by_name]
    if missing:
        raise ValueError(f"Missing required joints for preferred mapping: {missing}")
    return coordinates[:, [index_by_name[name] for name in PREFERRED_JOINTS], :]


def average_points(*points: np.ndarray) -> np.ndarray:
    stacked = np.stack(points, axis=0)
    return np.nanmean(stacked, axis=0)


def convert_mediapipe_to_preferred(coordinates: np.ndarray, joint_names: list[str]) -> np.ndarray:
    if joint_names != MEDIAPIPE_JOINTS:
        raise ValueError("MediaPipe conversion requires standard 33-landmark ordering")
    index = {name: idx for idx, name in enumerate(joint_names)}
    left_shoulder = coordinates[:, index["left_shoulder"], :]
    right_shoulder = coordinates[:, index["right_shoulder"], :]
    left_hip = coordinates[:, index["left_hip"], :]
    right_hip = coordinates[:, index["right_hip"], :]
    shoulder_mid = average_points(left_shoulder, right_shoulder)
    hip_mid = average_points(left_hip, right_hip)
    spine = average_points(shoulder_mid, hip_mid)

    derived = [
        coordinates[:, index["nose"], :],
        shoulder_mid,
        spine,
        left_shoulder,
        right_shoulder,
        coordinates[:, index["left_elbow"], :],
        coordinates[:, index["right_elbow"], :],
        coordinates[:, index["left_wrist"], :],
        coordinates[:, index["right_wrist"], :],
        left_hip,
        right_hip,
        coordinates[:, index["left_knee"], :],
        coordinates[:, index["right_knee"], :],
        coordinates[:, index["left_ankle"], :],
        coordinates[:, index["right_ankle"], :],
    ]
    return np.stack(derived, axis=1)


def resolve_target_path(project_root: Path, clip_id: str, pose_jobs_csv: Path) -> Path:
    pose_jobs_path = project_root / pose_jobs_csv
    if pose_jobs_path.exists():
        for row in read_csv(pose_jobs_path):
            if row["clip_id"] == clip_id:
                return project_root / row["pose_output_path"]
    standardized_root = project_root / "data" / "baseball" / "standardized_pose"
    candidates = sorted(standardized_root.glob(f"*/{clip_id}.npz"))
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not resolve a unique target payload path for clip_id={clip_id}")


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    target_path = resolve_target_path(project_root, args.clip_id, args.pose_jobs_csv)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Target payload does not exist: {target_path}. Create a template first with "
            f"create_baseball_pose_example.py --clip-id {args.clip_id}"
        )

    input_path = args.input_path.resolve()
    coordinates, joint_names = load_coordinates(input_path, args.coordinates_key, args.joint_names_key)
    input_metadata = load_input_metadata(input_path)
    pose_job = read_pose_job(project_root, args.pose_jobs_csv, args.clip_id)
    joint_names = parse_joint_names(args.joint_names) or joint_names

    if coordinates.ndim != 3:
        raise ValueError(f"Expected coordinates with shape (time, joints, dims), got {coordinates.shape}")
    if coordinates.shape[2] not in (2, 3):
        raise ValueError(f"Unsupported coordinate dims: {coordinates.shape[2]}")

    if joint_names is None:
        if coordinates.shape[1] != len(PREFERRED_JOINTS):
            raise ValueError("Joint names are required unless the input already has 15 preferred-order joints")
        ordered = coordinates
    elif joint_names == PREFERRED_JOINTS:
        ordered = coordinates
    elif joint_names == MEDIAPIPE_JOINTS:
        ordered = convert_mediapipe_to_preferred(coordinates, joint_names)
    else:
        ordered = reorder_to_preferred(coordinates, joint_names)

    template = np.load(target_path)
    metadata = json.loads(str(template["metadata"]))
    if pose_job is not None:
        metadata["clip_id"] = pose_job["clip_id"]
        metadata["source_name"] = pose_job["source_name"]
        metadata["task_type"] = pose_job["task_type"]
        metadata["view_type"] = pose_job.get("view_type", metadata.get("view_type", ""))
    metadata["coordinate_dims"] = int(ordered.shape[2])
    metadata["pose_source"] = args.pose_source
    inferred_fps = input_metadata.get("fps")
    metadata["fps"] = float(args.fps) if args.fps is not None else float(inferred_fps or metadata["fps"])
    metadata["dt"] = 1.0 / float(metadata["fps"])
    metadata["confidence_available"] = bool(metadata.get("confidence_available", False))
    metadata["joint_schema"] = "preferred_15_joint"
    metadata["tracking_notes"] = (
        "Coordinates imported with populate_baseball_pose_payload.py. "
        "Validate alignment against source video before downstream use."
    )
    quality_flags = [flag for flag in metadata.get("quality_flags", []) if flag != "template_payload"]
    metadata["quality_flags"] = quality_flags

    np.savez_compressed(
        target_path,
        coordinates=np.asarray(ordered, dtype=np.float32),
        joint_names=np.array(PREFERRED_JOINTS, dtype="<U32"),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(target_path.relative_to(project_root).as_posix())


if __name__ == "__main__":
    main()
