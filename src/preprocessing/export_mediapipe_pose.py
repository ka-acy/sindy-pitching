#!/usr/bin/env python3
"""Export MediaPipe Pose landmarks from a video into a compact NPZ payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MEDIAPIPE_JOINT_NAMES = [
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
    parser.add_argument("--input-video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output .npz path.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/pose_landmarker_full.task"),
        help="Path to MediaPipe pose landmarker task bundle.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help="MediaPipe pose model complexity.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe detection confidence threshold.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe tracking confidence threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_video = args.input_video.resolve()
    output_path = args.output_path.resolve()
    model_path = (Path.cwd() / args.model_path).resolve() if not args.model_path.is_absolute() else args.model_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model asset: {model_path}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    coordinates: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    detected_frames = 0
    backend = "mediapipe_tasks"

    def append_empty_frame() -> None:
        coordinates.append(np.full((33, 3), np.nan, dtype=np.float32))
        visibility.append(np.zeros(33, dtype=np.float32))

    try:
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=args.min_detection_confidence,
            min_pose_presence_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        pose = vision.PoseLandmarker.create_from_options(options)
        try:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(round((frame_index / fps) * 1000.0)) if fps > 0 else frame_index * 33
                result = pose.detect_for_video(mp_image, timestamp_ms)
                frame_index += 1
                if not result.pose_landmarks:
                    append_empty_frame()
                    continue

                detected_frames += 1
                frame_coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks[0]],
                    dtype=np.float32,
                )
                frame_visibility = np.array(
                    [lm.visibility for lm in result.pose_landmarks[0]],
                    dtype=np.float32,
                )
                coordinates.append(frame_coords)
                visibility.append(frame_visibility)
        finally:
            pose.close()
    except OSError:
        backend = "mediapipe_solutions_pose"
        cap.release()
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video for fallback backend: {input_video}")
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_frame)
                if not result.pose_landmarks:
                    append_empty_frame()
                    continue

                detected_frames += 1
                frame_coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                frame_visibility = np.array(
                    [lm.visibility for lm in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                coordinates.append(frame_coords)
                visibility.append(frame_visibility)
        finally:
            pose.close()
            cap.release()
    else:
        cap.release()

    coordinate_array = np.stack(coordinates, axis=0) if coordinates else np.empty((0, 33, 3), dtype=np.float32)
    visibility_array = np.stack(visibility, axis=0) if visibility else np.empty((0, 33), dtype=np.float32)

    metadata = {
        "input_video": str(input_video),
        "model_path": str(model_path),
        "fps": float(fps),
        "num_frames": int(frame_count or coordinate_array.shape[0]),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "pose_model": "mediapipe_pose",
        "pose_backend": backend,
        "model_complexity": args.model_complexity,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "detected_frames": detected_frames,
        "coordinate_space": "normalized_image_with_depth",
    }

    np.savez_compressed(
        output_path,
        coordinates=coordinate_array,
        visibility=visibility_array,
        joint_names=np.array(MEDIAPIPE_JOINT_NAMES, dtype="<U32"),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(output_path.as_posix())
    print(f"frames={coordinate_array.shape[0]} detected={detected_frames} fps={fps:.6f}")


if __name__ == "__main__":
    main()
