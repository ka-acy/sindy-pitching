#!/usr/bin/env python3
"""Extract biomechanical state vectors and derivatives from processed skeleton clips."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.signal import savgol_filter


ANGLE_FEATURES = {
    "trunk_rotation_angle",
    "shoulder_rotation_angle",
    "hip_rotation_angle",
    "hip_shoulder_separation_angle",
    "trunk_forward_lean_angle",
    "trunk_lateral_tilt_angle",
    "right_elbow_flexion_angle",
    "left_elbow_flexion_angle",
    "right_knee_flexion_angle",
    "left_knee_flexion_angle",
}


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
        help="Processed split to featurize.",
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


def load_processed_clip(path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    payload = np.load(path)
    coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    joint_names = payload["joint_names"].tolist()
    metadata = json.loads(str(payload["metadata"]))
    return coordinates, joint_names, metadata


def joint_index(joint_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(joint_names)}


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def projected_angle(vector: np.ndarray, first_axis: int, second_axis: int) -> np.ndarray:
    return np.arctan2(vector[:, second_axis], vector[:, first_axis])


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    numerator = np.sum(v1 * v2, axis=1)
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    denom = np.clip(denom, 1e-8, None)
    cosine = np.clip(numerator / denom, -1.0, 1.0)
    return np.arccos(cosine)


def velocity_magnitude(points: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.gradient(points, dt, axis=0)
    return np.linalg.norm(velocity, axis=1)


def pair_distance(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points_a - points_b, axis=1)


def ensure_coordinate_dims(coordinates: np.ndarray, minimum_dims: int) -> np.ndarray:
    if coordinates.ndim != 3:
        raise ValueError(f"Expected coordinates with shape (frames, joints, dims), got {coordinates.shape}")
    current_dims = coordinates.shape[2]
    if current_dims >= minimum_dims:
        return coordinates

    pad_width = minimum_dims - current_dims
    padding = np.zeros((coordinates.shape[0], coordinates.shape[1], pad_width), dtype=coordinates.dtype)
    return np.concatenate([coordinates, padding], axis=2)


def build_feature_series(coordinates: np.ndarray, joint_names: list[str], dt: float) -> dict[str, np.ndarray]:
    coordinates = ensure_coordinate_dims(coordinates, 3)
    idx = joint_index(joint_names)

    hip_center = 0.5 * (coordinates[:, idx["left_hip"], :] + coordinates[:, idx["right_hip"], :])
    trunk_vector = coordinates[:, idx["neck"], :] - hip_center
    hip_axis = coordinates[:, idx["left_hip"], :] - coordinates[:, idx["right_hip"], :]
    shoulder_axis = coordinates[:, idx["left_shoulder"], :] - coordinates[:, idx["right_shoulder"], :]

    right_upper = coordinates[:, idx["right_shoulder"], :] - coordinates[:, idx["right_elbow"], :]
    right_forearm = coordinates[:, idx["right_wrist"], :] - coordinates[:, idx["right_elbow"], :]
    left_upper = coordinates[:, idx["left_shoulder"], :] - coordinates[:, idx["left_elbow"], :]
    left_forearm = coordinates[:, idx["left_wrist"], :] - coordinates[:, idx["left_elbow"], :]
    right_thigh = coordinates[:, idx["right_hip"], :] - coordinates[:, idx["right_knee"], :]
    right_shank = coordinates[:, idx["right_ankle"], :] - coordinates[:, idx["right_knee"], :]
    left_thigh = coordinates[:, idx["left_hip"], :] - coordinates[:, idx["left_knee"], :]
    left_shank = coordinates[:, idx["left_ankle"], :] - coordinates[:, idx["left_knee"], :]

    trunk_rotation = projected_angle(trunk_vector, 0, 2)
    shoulder_rotation = projected_angle(shoulder_axis, 0, 2)
    hip_rotation = projected_angle(hip_axis, 0, 2)
    stride_ankle_separation = pair_distance(
        coordinates[:, idx["left_ankle"], :],
        coordinates[:, idx["right_ankle"], :],
    )

    return {
        "trunk_rotation_angle": trunk_rotation,
        "shoulder_rotation_angle": shoulder_rotation,
        "hip_rotation_angle": hip_rotation,
        "hip_shoulder_separation_angle": wrap_angle(shoulder_rotation - hip_rotation),
        "trunk_forward_lean_angle": np.arctan2(trunk_vector[:, 2], trunk_vector[:, 1]),
        "trunk_lateral_tilt_angle": np.arctan2(trunk_vector[:, 0], trunk_vector[:, 1]),
        "right_elbow_flexion_angle": angle_between(right_upper, right_forearm),
        "left_elbow_flexion_angle": angle_between(left_upper, left_forearm),
        "right_knee_flexion_angle": angle_between(right_thigh, right_shank),
        "left_knee_flexion_angle": angle_between(left_thigh, left_shank),
        "right_knee_lift": coordinates[:, idx["right_hip"], 1] - coordinates[:, idx["right_knee"], 1],
        "left_knee_lift": coordinates[:, idx["left_hip"], 1] - coordinates[:, idx["left_knee"], 1],
        "stride_ankle_separation": stride_ankle_separation,
        "right_wrist_speed": velocity_magnitude(coordinates[:, idx["right_wrist"], :], dt),
        "left_wrist_speed": velocity_magnitude(coordinates[:, idx["left_wrist"], :], dt),
    }


def adjusted_window(length: int, requested: int, polyorder: int) -> int:
    window = min(requested, length if length % 2 == 1 else length - 1)
    if window < 3 or window <= polyorder:
        return 0
    return window


def build_state_matrix(
    feature_series: dict[str, np.ndarray],
    feature_names: list[str],
    unwrap_angles: bool,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name in feature_names:
        series = np.asarray(feature_series[name], dtype=np.float64)
        if unwrap_angles and name in ANGLE_FEATURES:
            series = np.unwrap(series)
        columns.append(series)
    return np.column_stack(columns)


def estimate_derivatives(
    state_matrix: np.ndarray,
    dt: float,
    feature_names: list[str],
    unwrap_angles: bool,
    window_length: int,
    polyorder: int,
) -> tuple[np.ndarray, int]:
    matrix = state_matrix.copy()
    if unwrap_angles:
        for col_idx, name in enumerate(feature_names):
            if name in ANGLE_FEATURES:
                matrix[:, col_idx] = np.unwrap(matrix[:, col_idx])

    window = adjusted_window(matrix.shape[0], window_length, polyorder)
    if window == 0:
        return np.gradient(matrix, dt, axis=0), 0

    derivatives = savgol_filter(
        matrix,
        window_length=window,
        polyorder=polyorder,
        deriv=1,
        delta=dt,
        axis=0,
        mode="interp",
    )
    return derivatives, window


def save_feature_payload(
    path: Path,
    state_matrix: np.ndarray,
    derivative_matrix: np.ndarray,
    feature_names: list[str],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=state_matrix.astype(np.float32),
        X_dot=derivative_matrix.astype(np.float32),
        feature_names=np.array(feature_names, dtype="<U64"),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_config((project_root / args.config).resolve())

    processed_root = project_root / config["dataset"]["processed_root"] / args.split
    manifest_path = processed_root / "manifest.csv"
    split_rows = read_csv_rows(manifest_path)
    if args.limit is not None:
        split_rows = split_rows[: args.limit]

    feature_cfg = config["features"]
    feature_names = list(feature_cfg["state_vector"])
    feature_output_root = processed_root / feature_cfg["output_subdir"]
    log_dir = project_root / config["logging"]["log_dir"]

    manifest_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for row in split_rows:
        clip_path = project_root / row["filepath"]
        coordinates, joint_names, clip_meta = load_processed_clip(clip_path)
        dt = float(row["dt"])
        feature_series = build_feature_series(coordinates, joint_names, dt)
        state_matrix = build_state_matrix(
            feature_series,
            feature_names,
            bool(feature_cfg["angle_unwrap"]),
        )
        derivative_matrix, derivative_window = estimate_derivatives(
            state_matrix,
            dt,
            feature_names,
            bool(feature_cfg["angle_unwrap"]),
            int(feature_cfg["derivative"]["window_length"]),
            int(feature_cfg["derivative"]["polyorder"]),
        )

        output_path = feature_output_root / Path(row["filepath"]).name
        metadata = {
            **clip_meta,
            "feature_names": feature_names,
            "num_features": len(feature_names),
            "derivative_method": feature_cfg["derivative"]["method"],
            "derivative_window_length": derivative_window,
            "derivative_polyorder": int(feature_cfg["derivative"]["polyorder"]),
        }
        save_feature_payload(output_path, state_matrix, derivative_matrix, feature_names, metadata)

        manifest_rows.append(
            {
                "subset_label": row["subset_label"],
                "subject_id": row["subject_id"],
                "action_id": row["action_id"],
                "trial_id": row["trial_id"],
                "filepath": output_path.relative_to(project_root).as_posix(),
                "dt": dt,
                "num_frames": state_matrix.shape[0],
                "num_features": state_matrix.shape[1],
            }
        )
        log_rows.append(
            {
                "subset_label": row["subset_label"],
                "subject_id": int(row["subject_id"]),
                "action_id": int(row["action_id"]),
                "trial_id": int(row["trial_id"]),
                "source_clip": row["filepath"],
                "feature_file": output_path.relative_to(project_root).as_posix(),
                "dt": dt,
                "num_frames": int(state_matrix.shape[0]),
                "num_features": int(state_matrix.shape[1]),
                "derivative_window_length": derivative_window,
            }
        )

    write_csv(
        feature_output_root / "manifest.csv",
        ["subset_label", "subject_id", "action_id", "trial_id", "filepath", "dt", "num_frames", "num_features"],
        manifest_rows,
    )
    if config["logging"]["write_run_log"]:
        write_jsonl(log_dir / f"{args.split}_feature_run_log.jsonl", log_rows)

    print(f"feature trials: {len(manifest_rows)}")
    print(f"output root: {feature_output_root.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
