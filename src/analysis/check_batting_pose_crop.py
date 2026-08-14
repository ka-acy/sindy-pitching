#!/usr/bin/env python3
"""Second feasibility attempt for batting: crop the plate region, upsample, then run pose.

Full-frame pose on batting clips finds a usable batter on only 28.8 percent of frames, with a
71 px torso against 80-120 px for pitchers (`check_batting_pose_feasibility.py`). The batter is
simply too small in a 1280x720 centerfield frame, and MediaPipe frequently either misses him or
fuses him with the pitcher into a single skeleton.

This tests the obvious fix: crop a region around home plate, upsample it, and run pose on the
crop so the batter fills more of the input. The crop deliberately excludes the pitcher's mound,
which removes the fusion failure mode as well as the largest distractor.

Subject selection inside the crop is a different problem from the full frame. The pitcher is
gone, but the catcher and umpire remain. The batter is taken as the tallest detected pose: the
catcher crouches, and while the umpire stands, he is behind the catcher and typically shorter in
frame. That heuristic is reported rather than trusted -- annotated crops are written out so the
tracked subject can be confirmed by eye, which is what caught the error in the first attempt.

Gate for proceeding to feature design: batter-found rate above 0.85 and torso height comparable
to the pitching pipeline's 80-120 px.

Usage:
    python src/analysis/check_batting_pose_crop.py [--n-clips 20] [--crop 0.28,0.08,0.75,0.68]
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

from src.analysis.check_batting_pose_feasibility import load_batting, outcome_of

MODEL = PROJECT_ROOT / "data/models/pose_landmarker_full.task"
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-clips", type=int, default=20)
    p.add_argument("--crop", default="0.28,0.08,0.75,0.68",
                   help="Fractional crop x0,y0,x1,y1 covering the plate/batter region.")
    p.add_argument("--upscale", type=float, default=2.0)
    p.add_argument("--num-poses", type=int, default=3)
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--frames-out", default="results/analysis/batting_feasibility/crop_frames")
    p.add_argument("--output-csv", default="results/analysis/batting_feasibility/crop_summary.csv")
    p.add_argument("--save-frames-per-clip", type=int, default=1)
    p.add_argument("--media-root", default="",
                   help="Directory holding the .mp4 files. Defaults to the segmented tree; "
                        "point it at a local cache when that volume is unavailable.")
    p.add_argument("--clip-ids-file", default="",
                   help="Newline-separated clip ids to use verbatim instead of resampling. "
                        "Use the ids from the full-frame run so the comparison is on the same clips.")
    return p.parse_args()


def load_named(clip_ids: list[str], media_root: Path) -> list[dict]:
    """Load specific clips by id from a given media root, keeping their outcome labels."""
    manifest = PROJECT_ROOT / "data/baseball/metadata/clip_manifest.csv"
    wanted = set(clip_ids)
    by_id: dict[str, dict] = {}
    with manifest.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["clip_id"] not in wanted or r["task_type"] != "batting":
                continue
            try:
                notes = json.loads(r["notes"])
            except Exception:
                notes = {}
            path = media_root / f"{r['clip_id']}.mp4"
            if path.exists():
                by_id[r["clip_id"]] = {"clip_id": r["clip_id"], "path": path,
                                       "outcome": outcome_of(notes.get("labels", []))}
    return [by_id[i] for i in clip_ids if i in by_id]


def main() -> None:
    args = parse_args()
    if not MODEL.exists():
        raise SystemExit(f"missing pose model: {MODEL}")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    x0f, y0f, x1f, y1f = (float(v) for v in args.crop.split(","))
    if args.clip_ids_file:
        ids = [l.strip() for l in Path(args.clip_ids_file).read_text().split("\n") if l.strip()]
        root = Path(args.media_root) if args.media_root else None
        if root and not root.is_absolute():
            root = PROJECT_ROOT / root
        clips = load_named(ids, root or Path("/mnt/e/MML_RESEARCH/SEGMENTED/segmented"))
        print(f"using {len(clips)} of {len(ids)} named clips from {root}")
    else:
        clips = load_batting(args.n_clips, args.seed)
    if not clips:
        raise SystemExit("no batting clips found; check --media-root / --clip-ids-file")
    print(f"checking {len(clips)} batting clips, crop=({x0f},{y0f})-({x1f},{y1f}), "
          f"upscale={args.upscale}x")

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
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
        n_proc = n_found = 0
        vis_list, torso_list, pose_counts, saved = [], [], [], 0

        with mp_vision.PoseLandmarker.create_from_options(opts) as lm:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % args.frame_stride:
                    idx += 1
                    continue
                H, W = frame.shape[:2]
                x0, y0 = int(x0f * W), int(y0f * H)
                x1, y1 = int(x1f * W), int(y1f * H)
                crop = frame[y0:y1, x0:x1]
                if args.upscale != 1.0:
                    crop = cv2.resize(crop, None, fx=args.upscale, fy=args.upscale,
                                      interpolation=cv2.INTER_CUBIC)
                ch, cw = crop.shape[:2]
                n_proc += 1

                res = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                    int(idx / fps * 1000))
                poses = res.pose_landmarks or []
                pose_counts.append(len(poses))
                if poses:
                    # Batter = tallest pose. The catcher crouches; the umpire stands behind
                    # him and is shorter in frame from this camera.
                    def height(lms):
                        ys = [l.y for l in lms]
                        return max(ys) - min(ys)
                    best = max(poses, key=height)
                    n_found += 1
                    vis_list.append(float(np.mean([l.visibility for l in best])))
                    sh = np.mean([best[L_SHOULDER].y, best[R_SHOULDER].y])
                    hp = np.mean([best[L_HIP].y, best[R_HIP].y])
                    torso_list.append(float(abs(sh - hp) * ch))

                    if saved < args.save_frames_per_clip and idx > 30:
                        ann = crop.copy()
                        cols = [(0, 255, 0), (0, 128, 255), (255, 0, 255)]
                        for pi, lms in enumerate(poses):
                            c = cols[pi % len(cols)]
                            for l in lms:
                                cv2.circle(ann, (int(l.x * cw), int(l.y * ch)), 3, c, -1)
                            bx0 = int(min(l.x for l in lms) * cw)
                            by0 = int(min(l.y for l in lms) * ch)
                            bx1 = int(max(l.x for l in lms) * cw)
                            by1 = int(max(l.y for l in lms) * ch)
                            cv2.rectangle(ann, (bx0, by0), (bx1, by1), c, 2)
                            tag = "batter?" if lms is best else f"p{pi}"
                            cv2.putText(ann, tag, (bx0, max(by0 - 6, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
                        cv2.imwrite(str(frames_dir / f"{clip['clip_id']}_f{idx}.png"), ann)
                        saved += 1
                idx += 1
        cap.release()

        row = {
            "clip_id": clip["clip_id"],
            "outcome": clip["outcome"],
            "frames_processed": n_proc,
            "batter_found_rate": n_found / n_proc if n_proc else 0.0,
            "mean_poses_per_frame": float(np.mean(pose_counts)) if pose_counts else 0.0,
            "batter_mean_visibility": float(np.mean(vis_list)) if vis_list else 0.0,
            "batter_torso_px_mean": float(np.mean(torso_list)) if torso_list else 0.0,
        }
        rows.append(row)
        print(f"  [{ci}/{len(clips)}] {clip['clip_id']} {clip['outcome']:12s} "
              f"found={row['batter_found_rate']:.2f} poses/frame={row['mean_poses_per_frame']:.1f} "
              f"vis={row['batter_mean_visibility']:.2f} torso={row['batter_torso_px_mean']:.0f}px")

    out = PROJECT_ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    found = float(np.mean([r["batter_found_rate"] for r in rows]))
    torso = float(np.mean([r["batter_torso_px_mean"] for r in rows]))
    vis = float(np.mean([r["batter_mean_visibility"] for r in rows]))
    print(f"\n--- aggregate over {len(rows)} clips (cropped + upsampled) ---")
    print(f"batter found rate       {found:.3f}   (full-frame baseline was 0.288)")
    print(f"batter mean visibility  {vis:.3f}")
    print(f"batter torso height     {torso:.0f} px  (full-frame was 71px; pitching 80-120px)")
    gate = found > 0.85 and torso > 80
    print(f"\ngate (found > 0.85 and torso > 80px): {'PASS' if gate else 'FAIL'}")
    print(f"annotated crops: {frames_dir.relative_to(PROJECT_ROOT).as_posix()}")
    print("Confirm by eye which skeleton is tagged batter? before trusting these numbers.")


if __name__ == "__main__":
    main()
