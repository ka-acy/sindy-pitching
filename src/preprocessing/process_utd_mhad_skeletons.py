#!/usr/bin/env python3
"""Standardize and preprocess UTD-MHAD skeleton trajectories.

This script implements the Step 3 / Step 4 foundation of the project:
- map raw Kinect joints into a standardized schema
- remove global translation via pelvis-centered coordinates
- optionally normalize orientation and subject scale
- interpolate short missing-value gaps
- smooth trajectories with Savitzky-Golay filtering
- trim clips to motion-relevant windows

Processed trajectories are written as `.npz` files under
`data/utd_mhad/processed/<split_name>/`, alongside a manifest and run log.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import yaml
from scipy.signal import savgol_filter


RAW_JOINT_NAMES = [
    "head",
    "shoulder_center",
    "spine",
    "hip_center",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
]

JOINT_ALIASES = {
    "neck": "shoulder_center",
    "torso": "spine",
}

DEFAULT_LOG_BASENAME = "run_log.jsonl"
DEFAULT_MANIFEST_BASENAME = "manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.utd_mhad.example.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--split",
        choices=("smoke_test", "proxy_subset"),
        default="smoke_test",
        help="Split manifest to preprocess.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for fast debugging.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_project_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    dataset_cfg = config["dataset"]
    return {
        "raw_root": project_root / dataset_cfg["raw_root"],
        "processed_root": project_root / dataset_cfg["processed_root"],
        "index_csv": project_root / dataset_cfg["index_csv"],
        "proxy_subset_csv": project_root / dataset_cfg["proxy_subset_csv"],
        "smoke_test_csv": project_root / dataset_cfg["smoke_test_csv"],
        "log_dir": project_root / config["logging"]["log_dir"],
    }


def build_index_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {
        (row["subject_id"], row["action_id"], row["trial_id"], row["modality"]): row
        for row in rows
    }


def load_raw_skeleton(path: Path) -> np.ndarray:
    mat = scipy.io.loadmat(path)
    raw = np.asarray(mat["d_skel"], dtype=np.float64)
    if raw.shape[0:2] != (20, 3):
        raise ValueError(f"Unexpected skeleton array shape for {path}: {raw.shape}")
    return np.transpose(raw, (2, 0, 1))


def normalize_joint_name(name: str) -> str:
    return JOINT_ALIASES.get(name, name)


def extract_target_joints(trajectory: np.ndarray, target_joints: list[str]) -> tuple[np.ndarray, list[str]]:
    raw_index = {name: idx for idx, name in enumerate(RAW_JOINT_NAMES)}
    joint_series = []
    standardized_names: list[str] = []
    for name in target_joints:
        raw_name = normalize_joint_name(name)
        if raw_name not in raw_index:
            raise KeyError(f"Unsupported target joint '{name}'")
        joint_series.append(trajectory[:, raw_index[raw_name], :])
        standardized_names.append(name)
    return np.stack(joint_series, axis=1), standardized_names


def interpolate_short_gaps(trajectory: np.ndarray, max_gap_frames: int) -> tuple[np.ndarray, int]:
    output = trajectory.copy()
    repaired_points = 0

    for joint_idx in range(output.shape[1]):
        for coord_idx in range(output.shape[2]):
            series = output[:, joint_idx, coord_idx]
            valid = np.isfinite(series)
            if valid.all():
                continue
            if not valid.any():
                raise ValueError("Encountered a coordinate series with no valid samples")

            invalid_idx = np.flatnonzero(~valid)
            splits = np.split(invalid_idx, np.where(np.diff(invalid_idx) != 1)[0] + 1)
            for block in splits:
                gap_len = len(block)
                start = block[0]
                end = block[-1]
                left = start - 1
                right = end + 1
                if left < 0 or right >= len(series) or gap_len > max_gap_frames:
                    raise ValueError(
                        f"Gap length {gap_len} exceeds interpolation limits "
                        f"(joint={joint_idx}, coord={coord_idx})"
                    )
                series[start : end + 1] = np.interp(
                    np.arange(start, end + 1),
                    np.array([left, right]),
                    np.array([series[left], series[right]]),
                )
                repaired_points += gap_len

    return output, repaired_points


def center_trajectory(trajectory: np.ndarray, joint_names: list[str], center_joint: str) -> np.ndarray:
    center_name = normalize_joint_name(center_joint)
    name_lookup = {normalize_joint_name(name): idx for idx, name in enumerate(joint_names)}

    if center_name in name_lookup:
        center = trajectory[:, name_lookup[center_name] : name_lookup[center_name] + 1, :]
    elif center_name == "hip_center":
        left_idx = name_lookup["left_hip"]
        right_idx = name_lookup["right_hip"]
        center = 0.5 * (trajectory[:, left_idx : left_idx + 1, :] + trajectory[:, right_idx : right_idx + 1, :])
    elif center_name == "shoulder_center":
        left_idx = name_lookup["left_shoulder"]
        right_idx = name_lookup["right_shoulder"]
        center = 0.5 * (trajectory[:, left_idx : left_idx + 1, :] + trajectory[:, right_idx : right_idx + 1, :])
    else:
        raise KeyError(f"Center joint '{center_joint}' is unavailable in the standardized schema")

    centered = trajectory - center
    return centered


def orientation_normalize(trajectory: np.ndarray, joint_names: list[str]) -> np.ndarray:
    name_lookup = {normalize_joint_name(name): idx for idx, name in enumerate(joint_names)}
    left_idx = name_lookup["left_hip"]
    right_idx = name_lookup["right_hip"]

    output = trajectory.copy()
    for frame_idx in range(output.shape[0]):
        pelvis_axis = output[frame_idx, left_idx] - output[frame_idx, right_idx]
        lateral = np.array([pelvis_axis[0], pelvis_axis[2]], dtype=np.float64)
        norm = np.linalg.norm(lateral)
        if norm < 1e-8:
            continue
        angle = np.arctan2(lateral[1], lateral[0])
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
        output[frame_idx, :, [0, 2]] = output[frame_idx, :, [0, 2]] @ rotation.T
    return output


def scale_normalize(trajectory: np.ndarray, joint_names: list[str]) -> tuple[np.ndarray, float]:
    name_lookup = {normalize_joint_name(name): idx for idx, name in enumerate(joint_names)}
    left_idx = name_lookup["left_shoulder"]
    right_idx = name_lookup["right_shoulder"]
    shoulder_width = np.linalg.norm(trajectory[:, left_idx] - trajectory[:, right_idx], axis=1)
    scale = float(np.median(shoulder_width))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Shoulder width scale is invalid")
    return trajectory / scale, scale


def smooth_trajectory(trajectory: np.ndarray, window_length: int, polyorder: int) -> tuple[np.ndarray, int]:
    frame_count = trajectory.shape[0]
    adjusted_window = min(window_length, frame_count if frame_count % 2 == 1 else frame_count - 1)
    if adjusted_window < 3 or adjusted_window <= polyorder:
        return trajectory.copy(), 0

    smoothed = savgol_filter(
        trajectory,
        window_length=adjusted_window,
        polyorder=polyorder,
        axis=0,
        mode="interp",
    )
    return smoothed, adjusted_window


def trim_motion_window(
    trajectory: np.ndarray,
    energy_threshold_ratio: float,
    padding_frames: int,
    min_frames: int,
) -> tuple[np.ndarray, int, int]:
    if trajectory.shape[0] <= min_frames:
        return trajectory.copy(), 0, trajectory.shape[0]

    displacement = np.linalg.norm(np.diff(trajectory, axis=0), axis=2)
    energy = displacement.mean(axis=1)
    if not np.any(energy > 0):
        return trajectory.copy(), 0, trajectory.shape[0]

    threshold = float(energy.max()) * energy_threshold_ratio
    active = np.flatnonzero(energy >= threshold)
    if active.size == 0:
        return trajectory.copy(), 0, trajectory.shape[0]

    start = max(int(active[0]) - padding_frames, 0)
    end = min(int(active[-1]) + 1 + padding_frames, trajectory.shape[0])
    if end - start < min_frames:
        deficit = min_frames - (end - start)
        grow_left = deficit // 2
        grow_right = deficit - grow_left
        start = max(start - grow_left, 0)
        end = min(end + grow_right, trajectory.shape[0])
        if end - start < min_frames:
            start = max(end - min_frames, 0)
            end = min(start + min_frames, trajectory.shape[0])

    return trajectory[start:end], start, end


def write_npz(
    path: Path,
    trajectory: np.ndarray,
    joint_names: list[str],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coordinates=trajectory.astype(np.float32),
        joint_names=np.array(joint_names, dtype="<U32"),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def process_trial(
    split_row: dict[str, str],
    skeleton_index_row: dict[str, str],
    config: dict[str, Any],
    project_root: Path,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    skeleton_path = project_root / skeleton_index_row["filepath"]
    fps = float(skeleton_index_row["fps"])
    dt = 1.0 / fps

    raw_traj = load_raw_skeleton(skeleton_path)
    target_joints = config["pose"]["target_joints"]
    trajectory, joint_names = extract_target_joints(raw_traj, target_joints)

    repaired_points = 0
    if config["preprocessing"]["interpolate_short_gaps"]:
        trajectory, repaired_points = interpolate_short_gaps(
            trajectory,
            int(config["preprocessing"]["max_gap_frames"]),
        )

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
        "subject_id": int(split_row["subject_id"]),
        "action_id": int(split_row["action_id"]),
        "trial_id": int(split_row["trial_id"]),
        "subset_label": split_row["subset_label"],
        "source_filepath": skeleton_index_row["filepath"],
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
    }
    return trajectory, joint_names, metadata


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_config((project_root / args.config).resolve())
    paths = resolve_project_paths(config, project_root)

    index_lookup = build_index_lookup(read_csv_rows(paths["index_csv"]))
    split_csv = paths[f"{args.split}_csv"]
    split_rows = read_csv_rows(split_csv)
    if args.limit is not None:
        split_rows = split_rows[: args.limit]

    output_root = paths["processed_root"] / args.split
    manifest_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for split_row in split_rows:
        key = (
            split_row["subject_id"],
            split_row["action_id"],
            split_row["trial_id"],
            "skeleton",
        )
        if key not in index_lookup:
            raise KeyError(f"Missing skeleton index entry for {key}")

        trajectory, joint_names, metadata = process_trial(
            split_row,
            index_lookup[key],
            config,
            project_root,
        )
        stem = (
            f"a{metadata['action_id']}_s{metadata['subject_id']}_t{metadata['trial_id']}"
            f"_{metadata['subset_label']}"
        )
        output_path = output_root / f"{stem}.npz"
        write_npz(output_path, trajectory, joint_names, metadata)

        manifest_rows.append(
            {
                "subset_label": metadata["subset_label"],
                "subject_id": metadata["subject_id"],
                "action_id": metadata["action_id"],
                "trial_id": metadata["trial_id"],
                "filepath": output_path.relative_to(project_root).as_posix(),
                "fps": metadata["fps"],
                "dt": metadata["dt"],
                "num_frames": metadata["processed_num_frames"],
                "num_joints": trajectory.shape[1],
            }
        )
        log_rows.append(metadata)

    write_csv(
        output_root / DEFAULT_MANIFEST_BASENAME,
        ["subset_label", "subject_id", "action_id", "trial_id", "filepath", "fps", "dt", "num_frames", "num_joints"],
        manifest_rows,
    )
    if config["logging"]["write_run_log"]:
        write_jsonl(
            paths["log_dir"] / f"{args.split}_{DEFAULT_LOG_BASENAME}",
            log_rows,
        )

    print(f"processed trials: {len(manifest_rows)}")
    print(f"output root: {output_root.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
