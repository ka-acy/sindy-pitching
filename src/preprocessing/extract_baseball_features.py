#!/usr/bin/env python3
"""Extract biomechanical state vectors from processed Stage B baseball clips."""

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

from src.preprocessing.extract_utd_mhad_features import (
    ANGLE_FEATURES,
    adjusted_window,
    build_feature_series,
    build_state_matrix,
    estimate_derivatives,
    load_processed_clip,
    save_feature_payload,
    write_csv,
    write_jsonl,
)
from scipy.signal import savgol_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.baseball.example.yaml"),
        help="YAML config path for Stage B baseball feature settings.",
    )
    parser.add_argument(
        "--split",
        default="smoke_test",
        help="Processed baseball split to featurize.",
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


def canonical_handedness(value: str, default: str) -> str:
    normalized = str(value or "").strip().upper()
    if normalized in {"R", "RIGHT"}:
        return "R"
    if normalized in {"L", "LEFT"}:
        return "L"
    fallback = str(default or "R").strip().upper()
    return "L" if fallback in {"L", "LEFT"} else "R"


def infer_pitching_handedness(
    feature_series: dict[str, Any],
    default: str,
) -> tuple[str, str]:
    right_speed = np.asarray(feature_series["right_wrist_speed"], dtype=np.float64)
    left_speed = np.asarray(feature_series["left_wrist_speed"], dtype=np.float64)
    right_score = float(np.nanpercentile(right_speed, 95))
    left_score = float(np.nanpercentile(left_speed, 95))
    if np.isclose(right_score, left_score, atol=1e-6, rtol=1e-4):
        return canonical_handedness("", default), "default_tie"
    return ("R", "inferred_wrist_speed") if right_score > left_score else ("L", "inferred_wrist_speed")


def canonicalize_feature_series(
    feature_series: dict[str, Any],
    handedness: str,
) -> dict[str, Any]:
    dominant_prefix = "right" if handedness == "R" else "left"
    trail_prefix = "left" if handedness == "R" else "right"
    lead_prefix = trail_prefix
    canonical = dict(feature_series)
    canonical["dominant_elbow_flexion_angle"] = feature_series[f"{dominant_prefix}_elbow_flexion_angle"]
    canonical["trail_elbow_flexion_angle"] = feature_series[f"{trail_prefix}_elbow_flexion_angle"]
    canonical["dominant_knee_flexion_angle"] = feature_series[f"{dominant_prefix}_knee_flexion_angle"]
    canonical["trail_knee_flexion_angle"] = feature_series[f"{trail_prefix}_knee_flexion_angle"]
    canonical["lead_knee_lift"] = feature_series[f"{lead_prefix}_knee_lift"]
    canonical["trail_knee_lift"] = feature_series[f"{dominant_prefix}_knee_lift"]
    canonical["stride_ankle_separation"] = feature_series["stride_ankle_separation"]
    canonical["dominant_wrist_speed"] = feature_series[f"{dominant_prefix}_wrist_speed"]
    canonical["trail_wrist_speed"] = feature_series[f"{trail_prefix}_wrist_speed"]

    # Mirror signed rotations for L-handed clips. The underlying axes are
    # built as left_joint - right_joint, so a lefty's torso/hip rotation
    # sweeps opposite a righty's. Without this flip, fitting a single
    # dynamical system across mixed handedness collapses the rotation
    # equations toward zero.
    if handedness == "L":
        for key in ("trunk_rotation_angle", "shoulder_rotation_angle", "hip_rotation_angle", "hip_shoulder_separation_angle", "trunk_lateral_tilt_angle"):
            if key in canonical:
                canonical[key] = -np.asarray(canonical[key], dtype=np.float64)
    return canonical


def compute_delay_state(
    state_matrix: np.ndarray,
    feature_names: list[str],
    depth: int,
    drop_features: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Augment the state with recent increments: [x(t), x(t)-x(t-1), ..., x(t-d+2)-x(t-d+1)].

    An autonomous first-order state of joint positions does not determine its own next
    increment, which is why the fitted equations score no better than persistence. Supplying
    two frames of history raises skill over persistence from about +0.01 to +0.38 on the same
    clips and the same target, and it is also the only state on which the threshold sweep
    discriminates between candidates at all.

    `drop_features` removes derived columns. `hip_shoulder_separation_angle` is shoulder
    rotation minus hip rotation, and the identity holds near-exactly on some clip subsets, so
    a training draw made mostly of those is rank-deficient in three columns and least squares
    answers with enormous cancelling coefficients. Removing it takes divergence from about one
    split in ten to none, at a cost of 0.0009 skill.

    Note that this state cannot be integrated forward as a dynamical system: the delay columns
    are not independent degrees of freedom. It is a one-step predictor, and reconstruction
    metrics computed from it are meaningless.
    """
    keep = [i for i, n in enumerate(feature_names) if n not in set(drop_features)]
    base = state_matrix[:, keep]
    names = [feature_names[i] for i in keep]
    if depth < 2:
        return base, names

    start = depth - 1
    if base.shape[0] <= start:
        return base[:0], names

    blocks = [base[start:]]
    out_names = list(names)
    for k in range(depth - 1):
        lo, hi = start - k, base.shape[0] - k
        blocks.append(base[lo:hi] - base[lo - 1:hi - 1])
        out_names += [f"d{k + 1}({n})" for n in names]
    return np.hstack(blocks), out_names


def compute_second_order_state(
    feature_series: dict[str, Any],
    position_names: list[str],
    dt: float,
    window_length: int,
    polyorder: int,
    unwrap_angles: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_pos, V, A) for a strict second-order state.

    Each output has shape (T, len(position_names)). A single Savitzky-Golay
    fit per position column is queried at deriv 0/1/2 so that X (positions +
    velocities) and X_dot (velocities + accelerations) stay internally
    consistent and share the same smoothing. Angle columns are unwrapped
    before filtering; rate columns carry no 2*pi discontinuity so they are
    not unwrapped.
    """
    pos_cols: list[np.ndarray] = []
    vel_cols: list[np.ndarray] = []
    acc_cols: list[np.ndarray] = []
    for name in position_names:
        series = np.asarray(feature_series[name], dtype=np.float64)
        if unwrap_angles and name in ANGLE_FEATURES:
            series = np.unwrap(series)
        window = adjusted_window(series.shape[0], window_length, polyorder)
        if window == 0:
            pos = series
            vel = np.gradient(series, dt)
            acc = np.gradient(vel, dt)
        else:
            pos = savgol_filter(series, window, polyorder, deriv=0, mode="interp")
            vel = savgol_filter(series, window, polyorder, deriv=1, delta=dt, mode="interp")
            acc = savgol_filter(series, window, polyorder, deriv=2, delta=dt, mode="interp")
        pos_cols.append(pos)
        vel_cols.append(vel)
        acc_cols.append(acc)
    return np.column_stack(pos_cols), np.column_stack(vel_cols), np.column_stack(acc_cols)


def zscore_series(series: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    scale = float(np.std(values))
    if scale < 1e-8:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / scale


def align_feature_series(
    feature_series: dict[str, Any],
    task_type: str,
    alignment_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not alignment_cfg.get("enabled", False):
        first_key = next(iter(feature_series))
        return feature_series, {"applied": False, "anchor_frame": "", "window_start": 0, "window_end": len(feature_series[first_key])}

    allowed_tasks = set(alignment_cfg.get("task_types", []))
    if allowed_tasks and task_type not in allowed_tasks:
        first_key = next(iter(feature_series))
        return feature_series, {"applied": False, "anchor_frame": "", "window_start": 0, "window_end": len(feature_series[first_key])}

    anchor_feature = str(alignment_cfg["anchor_feature"])
    anchor_scores = zscore_series(np.asarray(feature_series[anchor_feature], dtype=np.float64))
    secondary_feature = str(alignment_cfg.get("secondary_feature", "") or "")
    if secondary_feature:
        anchor_scores = anchor_scores + float(alignment_cfg.get("secondary_weight", 0.5)) * zscore_series(
            np.asarray(feature_series[secondary_feature], dtype=np.float64)
        )

    anchor_frame = int(np.argmax(anchor_scores))
    pre_frames = int(alignment_cfg.get("pre_frames", 0))
    post_frames = int(alignment_cfg.get("post_frames", 0))
    min_frames = int(alignment_cfg.get("min_frames", 1))
    total_frames = len(next(iter(feature_series.values())))
    start = max(0, anchor_frame - pre_frames)
    end = min(total_frames, anchor_frame + post_frames + 1)
    if end - start < min_frames:
        deficit = min_frames - (end - start)
        extend_left = min(start, deficit // 2)
        extend_right = min(total_frames - end, deficit - extend_left)
        start -= extend_left
        end += extend_right
        remaining = min_frames - (end - start)
        if remaining > 0:
            shift_left = min(start, remaining)
            start -= shift_left
            end = min(total_frames, end + (remaining - shift_left))

    aligned = {name: np.asarray(series)[start:end] for name, series in feature_series.items()}
    return aligned, {
        "applied": True,
        "anchor_feature": anchor_feature,
        "secondary_feature": secondary_feature,
        "anchor_frame": anchor_frame,
        "window_start": start,
        "window_end": end,
    }


def main() -> None:
    args = parse_args()
    config = load_config((PROJECT_ROOT / args.config).resolve())

    processed_root = PROJECT_ROOT / "data" / "baseball" / "processed" / args.split
    manifest_path = processed_root / "manifest.csv"
    split_rows = read_csv_rows(manifest_path)
    if args.limit is not None:
        split_rows = split_rows[: args.limit]

    feature_cfg = config["features"]
    feature_names = list(feature_cfg["state_vector"])
    default_handedness = str(feature_cfg.get("default_handedness", "R"))
    infer_handedness_for_pitching = bool(feature_cfg.get("infer_handedness_for_pitching", False))
    second_order = bool(feature_cfg.get("second_order", False))
    alignment_cfg = dict(feature_cfg.get("alignment", {}))
    feature_output_root = processed_root / feature_cfg["output_subdir"]
    log_dir = PROJECT_ROOT / config["logging"]["log_dir"]

    manifest_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for row in split_rows:
        clip_path = PROJECT_ROOT / row["filepath"]
        coordinates, joint_names, clip_meta = load_processed_clip(clip_path)
        dt = float(row["dt"])
        feature_series = build_feature_series(coordinates, joint_names, dt)
        metadata_handedness = str(clip_meta.get("handedness", "")).strip()
        if metadata_handedness:
            resolved_handedness = canonical_handedness(metadata_handedness, default_handedness)
            handedness_source = "metadata"
        elif row["task_type"] == "pitching" and infer_handedness_for_pitching:
            resolved_handedness, handedness_source = infer_pitching_handedness(feature_series, default_handedness)
        else:
            resolved_handedness = canonical_handedness("", default_handedness)
            handedness_source = "default"
        feature_series = canonicalize_feature_series(feature_series, resolved_handedness)
        feature_series, alignment_meta = align_feature_series(feature_series, row["task_type"], alignment_cfg)
        delay_cfg = feature_cfg.get("delay_embedding", {}) or {}
        use_delay = bool(delay_cfg.get("enabled", False))
        if use_delay:
            base_state = build_state_matrix(
                feature_series,
                feature_names,
                bool(feature_cfg["angle_unwrap"]),
            )
            state_matrix, output_feature_names = compute_delay_state(
                base_state,
                feature_names,
                int(delay_cfg.get("depth", 3)),
                list(delay_cfg.get("drop_features", [])),
            )
            # Angles were already unwrapped when the base state was built, and the increment
            # columns are differences rather than angles, so no further unwrapping applies.
            derivative_matrix, derivative_window = estimate_derivatives(
                state_matrix,
                dt,
                output_feature_names,
                False,
                int(feature_cfg["derivative"]["window_length"]),
                int(feature_cfg["derivative"]["polyorder"]),
            )
        elif second_order:
            position_names = feature_names
            window_length = int(feature_cfg["derivative"]["window_length"])
            polyorder = int(feature_cfg["derivative"]["polyorder"])
            X_pos, V, A = compute_second_order_state(
                feature_series,
                position_names,
                dt,
                window_length,
                polyorder,
                bool(feature_cfg["angle_unwrap"]),
            )
            state_matrix = np.hstack([X_pos, V])
            derivative_matrix = np.hstack([V, A])
            output_feature_names = list(position_names) + [f"{name}_rate" for name in position_names]
            derivative_window = adjusted_window(state_matrix.shape[0], window_length, polyorder)
        else:
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
            output_feature_names = feature_names

        output_path = feature_output_root / Path(row["filepath"]).name
        metadata = {
            **clip_meta,
            "resolved_handedness": resolved_handedness,
            "resolved_handedness_source": handedness_source,
            "alignment": alignment_meta,
            "feature_names": output_feature_names,
            "num_features": len(output_feature_names),
            "derivative_method": feature_cfg["derivative"]["method"],
            "derivative_window_length": derivative_window,
            "derivative_polyorder": int(feature_cfg["derivative"]["polyorder"]),
        }
        save_feature_payload(output_path, state_matrix, derivative_matrix, output_feature_names, metadata)

        manifest_rows.append(
            {
                "clip_id": row["clip_id"],
                "source_name": row["source_name"],
                "task_type": row["task_type"],
                "filepath": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "dt": dt,
                "num_frames": state_matrix.shape[0],
                "num_features": state_matrix.shape[1],
            }
        )
        log_rows.append(
            {
                "clip_id": row["clip_id"],
                "source_name": row["source_name"],
                "task_type": row["task_type"],
                "source_clip": row["filepath"],
                "feature_file": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "dt": dt,
                "resolved_handedness": resolved_handedness,
                "resolved_handedness_source": handedness_source,
                "alignment_applied": alignment_meta["applied"],
                "alignment_anchor_frame": alignment_meta["anchor_frame"],
                "num_frames": int(state_matrix.shape[0]),
                "num_features": int(state_matrix.shape[1]),
                "derivative_window_length": derivative_window,
            }
        )

    write_csv(
        feature_output_root / "manifest.csv",
        ["clip_id", "source_name", "task_type", "filepath", "dt", "num_frames", "num_features"],
        manifest_rows,
    )
    if config["logging"]["write_run_log"]:
        write_jsonl(log_dir / f"baseball_{args.split}_feature_run_log.jsonl", log_rows)

    print(f"feature clips: {len(manifest_rows)}")
    print(f"output root: {feature_output_root.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
