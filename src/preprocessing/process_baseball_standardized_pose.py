#!/usr/bin/env python3
"""Standardize and preprocess Stage B baseball pose trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.process_utd_mhad_skeletons import (
    DEFAULT_LOG_BASENAME,
    DEFAULT_MANIFEST_BASENAME,
    center_trajectory,
    interpolate_short_gaps,
    orientation_normalize,
    scale_normalize,
    smooth_trajectory,
    trim_motion_window,
    write_csv,
    write_jsonl,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.baseball.example.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--split",
        default="smoke_test",
        help="Output split name, e.g. smoke_test or train.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for fast debugging.",
    )
    parser.add_argument(
        "--task-type",
        default="",
        help="Optional task filter, e.g. pitching or batting.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_split_manifest_path(project_root: Path, split: str, task_type: str) -> Path | None:
    splits_root = project_root / "data" / "baseball" / "splits"
    candidates: list[str] = []
    normalized_task = task_type.strip().lower()

    if split == "smoke_test":
        candidates.append("smoke_test_local.csv")
    if normalized_task:
        candidates.extend(
            [
                f"{normalized_task}_local_{split}.csv",
                f"{normalized_task}_{split}.csv",
            ]
        )
    else:
        candidates.extend(
            [
                f"pitching_local_{split}.csv",
                f"batting_local_{split}.csv",
            ]
        )
    candidates.append(f"{split}.csv")

    for candidate in candidates:
        path = splits_root / candidate
        if path.exists():
            return path
    return None


def load_pose_clip(path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    payload = np.load(path)
    coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    joint_names = payload["joint_names"].tolist()
    metadata = json.loads(str(payload["metadata"]))
    return coordinates, joint_names, metadata


def is_placeholder_pose_payload(coordinates: np.ndarray, metadata: dict[str, Any]) -> bool:
    quality_flags = {str(flag) for flag in metadata.get("quality_flags", [])}
    if "template_payload" in quality_flags:
        return True
    finite = coordinates[np.isfinite(coordinates)]
    if finite.size == 0:
        return True
    return float(np.nanstd(finite)) < 1e-8


def fill_edge_gaps(trajectory: np.ndarray) -> tuple[np.ndarray, int]:
    output = trajectory.copy()
    repaired_points = 0
    for joint_idx in range(output.shape[1]):
        for coord_idx in range(output.shape[2]):
            series = output[:, joint_idx, coord_idx]
            valid = np.isfinite(series)
            if not valid.any():
                raise ValueError(f"No valid samples for joint={joint_idx}, coord={coord_idx}")
            first_valid = int(np.flatnonzero(valid)[0])
            last_valid = int(np.flatnonzero(valid)[-1])
            if first_valid > 0:
                series[:first_valid] = series[first_valid]
                repaired_points += first_valid
            if last_valid < len(series) - 1:
                series[last_valid + 1 :] = series[last_valid]
                repaired_points += len(series) - 1 - last_valid
    return output, repaired_points


def interpolate_internal_gaps(trajectory: np.ndarray) -> tuple[np.ndarray, int]:
    output = trajectory.copy()
    repaired_points = 0
    for joint_idx in range(output.shape[1]):
        for coord_idx in range(output.shape[2]):
            series = output[:, joint_idx, coord_idx]
            valid = np.isfinite(series)
            if valid.all():
                continue
            if not valid.any():
                raise ValueError(f"No valid samples for joint={joint_idx}, coord={coord_idx}")

            invalid_idx = np.flatnonzero(~valid)
            splits = np.split(invalid_idx, np.where(np.diff(invalid_idx) != 1)[0] + 1)
            for block in splits:
                start = int(block[0])
                end = int(block[-1])
                left = start - 1
                right = end + 1
                if left < 0 or right >= len(series):
                    raise ValueError(f"Unbounded gap after edge fill for joint={joint_idx}, coord={coord_idx}")
                series[start : end + 1] = np.interp(
                    np.arange(start, end + 1),
                    np.array([left, right]),
                    np.array([series[left], series[right]]),
                )
                repaired_points += len(block)
    return output, repaired_points


def process_clip(
    clip_row: dict[str, str],
    config: dict[str, Any],
    project_root: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    clip_path = project_root / clip_row["filepath"]
    raw_traj, joint_names, raw_meta = load_pose_clip(clip_path)
    if is_placeholder_pose_payload(raw_traj, raw_meta):
        raise ValueError(f"Placeholder pose payload: {clip_row['clip_id']}")
    fps = float(clip_row["fps"])
    dt = float(clip_row["dt"])

    trajectory, repaired_points = fill_edge_gaps(raw_traj)
    if config["preprocessing"]["interpolate_short_gaps"]:
        trajectory, interior_repairs = interpolate_internal_gaps(trajectory)
        repaired_points += interior_repairs

    trajectory = center_trajectory(trajectory, joint_names, config["pose"]["center_joint"])

    if config["pose"]["orientation_normalization"]:
        trajectory = orientation_normalize(trajectory, joint_names)

    scale_factor = 1.0
    if config["preprocessing"]["scale_normalization"]:
        trajectory, scale_factor = scale_normalize(trajectory, joint_names)

    smoothed, smoothing_window = smooth_trajectory(
        trajectory,
        int(config["preprocessing"]["smoothing"]["window_length"]),
        int(config["preprocessing"]["smoothing"]["polyorder"]),
    )
    trajectory = smoothed

    trim_start = 0
    trim_end = trajectory.shape[0]
    if config["preprocessing"]["trim_motion_windows"]:
        trajectory, trim_start, trim_end = trim_motion_window(
            trajectory,
            float(config["preprocessing"]["trim"]["energy_threshold_ratio"]),
            int(config["preprocessing"]["trim"]["padding_frames"]),
            int(config["preprocessing"]["trim"]["min_frames"]),
        )

    metadata = {
        **raw_meta,
        "clip_id": clip_row["clip_id"],
        "source_name": clip_row["source_name"],
        "task_type": clip_row["task_type"],
        "source_filepath": clip_row["filepath"],
        "raw_num_frames": int(raw_traj.shape[0]),
        "processed_num_frames": int(trajectory.shape[0]),
        "fps": fps,
        "dt": dt,
        "center_joint": config["pose"]["center_joint"],
        "orientation_normalization": bool(config["pose"]["orientation_normalization"]),
        "scale_normalization": bool(config["preprocessing"]["scale_normalization"]),
        "scale_factor": scale_factor,
        "repaired_points": repaired_points,
        "savgol_window_length": smoothing_window,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "processed_split": config["dataset"]["default_split"],
    }
    return trajectory, joint_names, metadata


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    config = load_config((project_root / args.config).resolve())
    config["dataset"]["default_split"] = args.split

    manifest_path = project_root / config["dataset"]["standardized_pose_manifest"]
    processed_root = project_root / config["dataset"]["processed_root"] / args.split
    log_dir = project_root / config["logging"]["log_dir"]

    manifest_rows = read_csv_rows(manifest_path)
    manifest_rows = [row for row in manifest_rows if row["registered_clip_id"] == "True"]
    split_manifest_path = resolve_split_manifest_path(project_root, args.split, args.task_type)
    if split_manifest_path is not None:
        split_rows = read_csv_rows(split_manifest_path)
        split_clip_ids = {row["clip_id"] for row in split_rows}
        manifest_rows = [row for row in manifest_rows if row["clip_id"] in split_clip_ids]
    if args.task_type:
        manifest_rows = [row for row in manifest_rows if row["task_type"] == args.task_type]
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    output_manifest_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for row in manifest_rows:
        try:
            trajectory, joint_names, metadata = process_clip(row, config, project_root)
        except ValueError as exc:
            log_rows.append(
                {
                    "clip_id": row["clip_id"],
                    "source_name": row["source_name"],
                    "task_type": row["task_type"],
                    "source_filepath": row["filepath"],
                    "skipped": True,
                    "skip_reason": str(exc),
                }
            )
            continue
        stem = f"{row['clip_id']}_{row['task_type']}"
        output_path = processed_root / f"{stem}.npz"
        write_npz(output_path, trajectory, joint_names, metadata)

        output_manifest_rows.append(
            {
                "clip_id": row["clip_id"],
                "source_name": row["source_name"],
                "task_type": row["task_type"],
                "filepath": output_path.relative_to(project_root).as_posix(),
                "fps": metadata["fps"],
                "dt": metadata["dt"],
                "num_frames": metadata["processed_num_frames"],
                "num_joints": trajectory.shape[1],
            }
        )
        log_rows.append(metadata)

    write_csv(
        processed_root / DEFAULT_MANIFEST_BASENAME,
        ["clip_id", "source_name", "task_type", "filepath", "fps", "dt", "num_frames", "num_joints"],
        output_manifest_rows,
    )
    if config["logging"]["write_run_log"]:
        write_jsonl(log_dir / f"baseball_{args.split}_{DEFAULT_LOG_BASENAME}", log_rows)

    print(f"processed clips: {len(output_manifest_rows)}")
    print(f"output root: {processed_root.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
