#!/usr/bin/env python3
"""Go/no-go pose-quality check for batting clips before any feature design.

The pitching pipeline works because the pitcher is central, large, and moving roughly
perpendicular to the centerfield camera. A batter from the same camera is smaller, further
away, partly occluded by the catcher and umpire, and swings largely along the camera axis.
Two things have to hold before batting is worth pursuing:

1. A pose is detected on most frames.
2. The pose that gets tracked is the BATTER, not the pitcher, catcher, or umpire.

Point 2 is the real risk and the existing exporter cannot answer it: it requests a single
pose and takes `pose_landmarks[0]`, so on a batting frame it may silently lock onto the
pitcher. This script requests several poses per frame, records each one's bounding box and
mean landmark visibility, and writes annotated sample frames so the tracked subject can be
confirmed by eye rather than assumed.

Batter identification heuristic used for reporting only: among detected poses, the batter is
expected in the lower half of the frame and off-centre horizontally, whereas the pitcher sits
near the frame centre. The heuristic is reported alongside the raw per-pose geometry so it
can be checked rather than trusted.

Usage:
    python src/analysis/check_batting_pose_feasibility.py [--n-clips 20] [--frames-out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEG_ROOT = Path("/mnt/e/MML_RESEARCH/SEGMENTED/segmented")
MANIFEST = PROJECT_ROOT / "data/baseball/metadata/clip_manifest.csv"
MODEL = PROJECT_ROOT / "data/models/pose_landmarker_full.task"

# MediaPipe BlazePose landmark indices used for geometry.
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-clips", type=int, default=20)
    p.add_argument("--num-poses", type=int, default=4, help="Poses MediaPipe may return per frame.")
    p.add_argument("--frame-stride", type=int, default=3, help="Process every Nth frame.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--frames-out", default="results/analysis/batting_feasibility/frames")
    p.add_argument("--output-csv", default="results/analysis/batting_feasibility/summary.csv")
    p.add_argument("--save-frames-per-clip", type=int, default=1)
    return p.parse_args()


def outcome_of(labels: list[str]) -> str:
    """Three swing outcomes derivable from the MLB-YouTube segmented labels."""
    s = set(labels)
    if "swing" not in s:
        return "no_swing"
    if "in play" in s:
        return "in_play"
    if "foul" in s:
        return "foul"
    if "hit" not in s:
        return "swing_miss"
    return "contact_other"


def load_batting(n: int, seed: int) -> list[dict]:
    with MANIFEST.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["task_type"] == "batting"]
    out = []
    for r in rows:
        path = SEG_ROOT / f"{r['clip_id']}.mp4"
        if not path.exists():
            continue
        try:
            notes = json.loads(r["notes"])
        except Exception:
            continue
        out.append({"clip_id": r["clip_id"], "path": path,
                    "outcome": outcome_of(notes.get("labels", []))})
    # Stratify across outcomes so the check is not dominated by one class.
    by: dict[str, list[dict]] = {}
    for c in out:
        by.setdefault(c["outcome"], []).append(c)
    rng = random.Random(seed)
    picked: list[dict] = []
    classes = sorted(by)
    while len(picked) < n and any(by[c] for c in classes):
        for c in classes:
            if by[c] and len(picked) < n:
                picked.append(by[c].pop(rng.randrange(len(by[c]))))
    return picked


def main() -> None:
    args = parse_args()
    if not MODEL.exists():
        raise SystemExit(f"missing pose model: {MODEL}")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    clips = load_batting(args.n_clips, args.seed)
    if not clips:
        raise SystemExit("no batting clips with local media found")
    print(f"checking {len(clips)} batting clips")

    frames_dir = PROJECT_ROOT / args.frames_out
    frames_dir.mkdir(parents=True, exist_ok=True)

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = []
    for ci, clip in enumerate(clips, 1):
        cap = cv2.VideoCapture(str(clip["path"]))
        if not cap.isOpened():
            print(f"  [{ci}/{len(clips)}] {clip['clip_id']}: could not open")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
        n_processed = n_detected = 0
        pose_counts, batter_vis, batter_height, saved = [], [], [], 0
        n_batter = n_batter_missing = 0

        with mp_vision.PoseLandmarker.create_from_options(opts) as landmarker:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % args.frame_stride:
                    idx += 1
                    continue
                h, w = frame.shape[:2]
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = landmarker.detect_for_video(mp_img, int(idx / fps * 1000))
                n_processed += 1

                poses = res.pose_landmarks or []
                pose_counts.append(len(poses))
                if poses:
                    n_detected += 1
                    # Subject selection, corrected. From centerfield the PITCHER is the
                    # pose lowest in frame -- he is nearest the camera on the mound -- so
                    # scoring by "lowest" selects the pitcher, not the batter. The batter
                    # stands further away and so sits HIGHER in the image. Take the lowest
                    # pose as the pitcher and the batter as the next-lowest pose that does
                    # not substantially overlap him horizontally.
                    def box(lms):
                        xs = [lm.x for lm in lms]
                        ys = [lm.y for lm in lms]
                        return min(xs), min(ys), max(xs), max(ys)

                    boxed = sorted(((box(l), l) for l in poses), key=lambda t: -t[0][3])
                    pitcher_box = boxed[0][0]
                    best = None
                    for bx, lms in boxed[1:]:
                        overlap = max(0.0, min(bx[2], pitcher_box[2]) - max(bx[0], pitcher_box[0]))
                        width = max(bx[2] - bx[0], 1e-6)
                        if overlap / width < 0.6:
                            best = lms
                            break
                    if best is None:
                        n_batter_missing += 1
                        idx += 1
                        continue
                    n_batter += 1
                    vis = float(np.mean([lm.visibility for lm in best]))
                    batter_vis.append(vis)
                    sh = np.mean([[best[L_SHOULDER].x, best[L_SHOULDER].y],
                                  [best[R_SHOULDER].x, best[R_SHOULDER].y]], axis=0)
                    hp = np.mean([[best[L_HIP].x, best[L_HIP].y],
                                  [best[R_HIP].x, best[R_HIP].y]], axis=0)
                    batter_height.append(float(abs(sh[1] - hp[1]) * h))

                    if saved < args.save_frames_per_clip and len(poses) >= 1 and idx > 30:
                        annotated = frame.copy()
                        colors = [(0, 255, 0), (0, 128, 255), (255, 0, 255), (255, 255, 0)]
                        for pi, lms in enumerate(poses):
                            col = colors[pi % len(colors)]
                            for lm in lms:
                                cv2.circle(annotated, (int(lm.x * w), int(lm.y * h)), 3, col, -1)
                            x0 = int(min(lm.x for lm in lms) * w)
                            y0 = int(min(lm.y for lm in lms) * h)
                            x1 = int(max(lm.x for lm in lms) * w)
                            y1 = int(max(lm.y for lm in lms) * h)
                            cv2.rectangle(annotated, (x0, y0), (x1, y1), col, 2)
                            cv2.putText(annotated, f"p{pi}", (x0, max(y0 - 6, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                        cv2.imwrite(str(frames_dir / f"{clip['clip_id']}_f{idx}.png"), annotated)
                        saved += 1
                idx += 1
        cap.release()

        row = {
            "clip_id": clip["clip_id"],
            "outcome": clip["outcome"],
            "frames_processed": n_processed,
            "detection_rate": n_detected / n_processed if n_processed else 0.0,
            "mean_poses_per_frame": float(np.mean(pose_counts)) if pose_counts else 0.0,
            "max_poses_in_any_frame": int(np.max(pose_counts)) if pose_counts else 0,
            "batter_mean_visibility": float(np.mean(batter_vis)) if batter_vis else 0.0,
            "batter_torso_px_mean": float(np.mean(batter_height)) if batter_height else 0.0,
            "batter_found_rate": n_batter / n_processed if n_processed else 0.0,
        }
        rows.append(row)
        print(f"  [{ci}/{len(clips)}] {clip['clip_id']} {clip['outcome']:14s} "
              f"det={row['detection_rate']:.2f} poses/frame={row['mean_poses_per_frame']:.1f} "
              f"vis={row['batter_mean_visibility']:.2f} torso={row['batter_torso_px_mean']:.0f}px "
              f"batter_found={row['batter_found_rate']:.2f}")

    out = PROJECT_ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    det = float(np.mean([r["detection_rate"] for r in rows]))
    vis = float(np.mean([r["batter_mean_visibility"] for r in rows]))
    torso = float(np.mean([r["batter_torso_px_mean"] for r in rows]))
    ppf = float(np.mean([r["mean_poses_per_frame"] for r in rows]))
    print(f"\n--- aggregate over {len(rows)} clips ---")
    print(f"detection rate          {det:.3f}")
    print(f"poses per frame         {ppf:.2f}   (>1 means subject selection is required)")
    bf = float(np.mean([r["batter_found_rate"] for r in rows]))
    print(f"BATTER found rate       {bf:.3f}   (a second, non-overlapping pose exists)")
    print(f"batter mean visibility  {vis:.3f}")
    print(f"batter torso height     {torso:.0f} px  (pitching torsos are ~80-120px for reference)")
    print(f"\nannotated frames: {frames_dir.relative_to(PROJECT_ROOT).as_posix()}")
    print("Inspect those frames before any go decision: the numbers cannot tell you")
    print("whether the tracked skeleton is the batter or the pitcher.")


if __name__ == "__main__":
    main()
