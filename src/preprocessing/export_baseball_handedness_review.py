#!/usr/bin/env python3
"""Export a handedness review packet for Stage B baseball clips."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FRAME_FRACTIONS = (0.15, 0.4, 0.65, 0.85)
FRAME_HEIGHT = 216


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/baseball/splits/pitching_local_pool.csv"),
        help="Clip list CSV relative to project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/review/baseball_handedness"),
        help="Output review directory relative to project root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of clips to export.",
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


def resize_frame(frame: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Invalid frame shape")
    scale = target_height / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def annotate_frame(frame: np.ndarray, label: str) -> np.ndarray:
    annotated = frame.copy()
    cv2.rectangle(annotated, (0, 0), (150, 28), (15, 15, 15), thickness=-1)
    cv2.putText(
        annotated,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def extract_preview(contact_sheet_path: Path, video_path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

        tiles: list[np.ndarray] = []
        for fraction in FRAME_FRACTIONS:
            frame_idx = min(frame_count - 1, max(0, int(round((frame_count - 1) * fraction))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
            tile = resize_frame(frame, FRAME_HEIGHT)
            tile = annotate_frame(tile, f"{fraction:.2f}")
            tiles.append(tile)

        panel = cv2.hconcat(tiles)
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(contact_sheet_path), panel):
            raise RuntimeError(f"Could not write preview image: {contact_sheet_path}")
        return frame_count, fps
    finally:
        cap.release()


def build_html(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        preview_rel = html.escape(row["preview_path"])
        clip_id = html.escape(row["clip_id"])
        pitch_type = html.escape(str(row.get("pitch_type", "")))
        speed = html.escape(str(row.get("pitch_speed_mph", "")))
        url = html.escape(str(row.get("video_url", "")))
        local_media_path = html.escape(str(row.get("local_media_path", "")))
        cards.append(
            f"""
<article class="card">
  <h2>{clip_id}</h2>
  <img src="{preview_rel}" alt="Preview for {clip_id}">
  <p><strong>Pitch:</strong> {pitch_type} {speed}</p>
  <p><strong>Frames:</strong> {row.get("num_frames", "")} at {row.get("fps", "")} fps</p>
  <p><strong>Label:</strong> fill <code>data/baseball/metadata/handedness_overrides.csv</code></p>
  <p><strong>Video URL:</strong> <a href="{url}">{url}</a></p>
  <p><strong>Local file:</strong> {local_media_path}</p>
</article>
""".strip()
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Baseball Handedness Review</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f5f1e8; color: #1f2328; }}
    h1 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 20px; }}
    .card {{ background: #fffdf8; border: 1px solid #d8d1c2; padding: 14px; border-radius: 12px; }}
    .card img {{ width: 100%; border-radius: 8px; border: 1px solid #d8d1c2; }}
    .card p {{ margin: 8px 0; line-height: 1.35; }}
    code {{ background: #f0eadf; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Baseball Handedness Review</h1>
  <p>Use these preview panels to fill <code>data/baseball/metadata/handedness_overrides.csv</code> with <code>R</code> or <code>L</code>.</p>
  <div class="grid">
    {' '.join(cards)}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    project_root = args.root.resolve()
    input_rows = read_csv(project_root / args.input_csv)
    if args.limit is not None:
        input_rows = input_rows[: args.limit]

    output_dir = project_root / args.output_dir
    preview_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, Any]] = []
    for row in input_rows:
        video_path = Path(row["local_media_path"])
        preview_path = preview_dir / f"{row['clip_id']}.jpg"
        num_frames, fps = extract_preview(preview_path, video_path)
        review_rows.append(
            {
                "clip_id": row["clip_id"],
                "task_type": row.get("task_type", ""),
                "video_url": row.get("video_url", ""),
                "pitch_type": row.get("pitch_type", ""),
                "pitch_speed_mph": row.get("pitch_speed_mph", ""),
                "local_media_path": row.get("local_media_path", ""),
                "preview_path": preview_path.relative_to(output_dir).as_posix(),
                "num_frames": num_frames,
                "fps": round(fps, 6),
            }
        )

    write_csv(
        output_dir / "review_manifest.csv",
        ["clip_id", "task_type", "video_url", "pitch_type", "pitch_speed_mph", "local_media_path", "preview_path", "num_frames", "fps"],
        review_rows,
    )
    (output_dir / "index.html").write_text(build_html(review_rows), encoding="utf-8")

    print(f"review clips: {len(review_rows)}")
    print(f"output: {output_dir.relative_to(project_root).as_posix()}")


if __name__ == "__main__":
    main()
