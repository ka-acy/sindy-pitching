#!/usr/bin/env python3
"""Ground truth for batter selection: render candidate tracks, then score rules against labels.

Five selection heuristics have failed in this workstream -- lowest pose in frame, tallest pose in
crop, mean wrist speed, track coverage, and a two-pass crop that inherits pass one's choice. Each
failure was caught only by opening a rendered frame and looking, and the aggregate coverage metric
could not detect any of them because it counts whether *a* pose was found, not whether it was the
right person.

So this stops proposing heuristics and makes selection measurable. Two modes:

  render   for each clip, build tracks and write a montage showing every track with a stable
           colour and ID across three frames, plus a CSV of per-track features. A human reads
           the montages once and records which track ID is the batter.

  score    given those labels, evaluate candidate selection rules by accuracy against them.

Labelling once and scoring many rules is the point: the eyeballing happens a fixed number of
times up front instead of silently, one frame at a time, every time a rule changes.

Usage:
    python src/analysis/label_batter_tracks.py render --n-clips 10
    # fill in results/analysis/batter_labeling/labels.csv, then:
    python src/analysis/label_batter_tracks.py score
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.check_batting_pose_feasibility import load_batting
from src.analysis.check_batting_track_selection import (
    L_HIP, L_SHOULDER, L_WRIST, MODEL, R_HIP, R_SHOULDER, R_WRIST,
    build_tracks, centroid, load_named, pan_onset, wrist_point,
)

OUT_DIR = PROJECT_ROOT / "results/analysis/batter_labeling"
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 255), (255, 255, 0), (0, 0, 255), (255, 128, 0)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("render", "score"))
    p.add_argument("--n-clips", type=int, default=10)
    p.add_argument("--crop", default="0.28,0.08,0.75,0.68")
    p.add_argument("--upscale", type=float, default=2.0)
    p.add_argument("--num-poses", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--match-radius", type=float, default=0.12)
    p.add_argument("--min-coverage", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--media-root", default="")
    p.add_argument("--clip-ids-file", default="")
    return p.parse_args()


def track_features(track: dict, n_frames: int, crop_h: int) -> dict:
    """Per-track descriptors that a selection rule could use."""
    cents = np.asarray(track["centroids"])
    wr = np.asarray(track["wrists"])
    heights, torsos = [], []
    for fi in track["frames"]:
        lms = track["landmarks"][fi]
        ys = [l.y for l in lms]
        heights.append(max(ys) - min(ys))
        sh = np.mean([lms[L_SHOULDER].y, lms[R_SHOULDER].y])
        hp = np.mean([lms[L_HIP].y, lms[R_HIP].y])
        torsos.append(abs(sh - hp) * crop_h)
    speed = float(np.mean(np.linalg.norm(np.diff(wr, axis=0), axis=1))) if wr.shape[0] > 1 else 0.0
    # Range, unlike speed, is not inflated by per-frame detection jitter: a swing sweeps a wide
    # arc, while a catcher's or umpire's hands stay in a small region however noisy the landmarks.
    wrist_range = float(np.linalg.norm(wr.max(axis=0) - wr.min(axis=0))) if wr.shape[0] > 1 else 0.0
    return {
        "coverage": len(track["frames"]) / max(n_frames, 1),
        "centroid_x": float(cents[:, 0].mean()),
        "centroid_y": float(cents[:, 1].mean()),
        "bbox_height": float(np.mean(heights)),
        "torso_px": float(np.mean(torsos)),
        "wrist_speed": speed,
        "wrist_range": wrist_range,
        "centroid_travel": float(np.linalg.norm(cents.max(axis=0) - cents.min(axis=0))),
    }


def render(args) -> None:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    x0f, y0f, x1f, y1f = (float(v) for v in args.crop.split(","))
    if args.clip_ids_file:
        ids = [l.strip() for l in Path(args.clip_ids_file).read_text().split("\n") if l.strip()]
        root = Path(args.media_root) if args.media_root else Path("/mnt/e/MML_RESEARCH/SEGMENTED/segmented")
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        clips = load_named(ids, root)[: args.n_clips]
    else:
        clips = load_batting(args.n_clips, args.seed)
    if not clips:
        raise SystemExit("no clips found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO, num_poses=args.num_poses,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    rows = []
    for ci, clip in enumerate(clips, 1):
        stop_at = pan_onset(clip["path"], 3.0)
        cap = cv2.VideoCapture(str(clip["path"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
        collected, frames = [], {}
        with mp_vision.PoseLandmarker.create_from_options(opts) as lm:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or idx >= stop_at:
                    break
                if idx % args.frame_stride:
                    idx += 1
                    continue
                H, W = frame.shape[:2]
                crop = frame[int(y0f * H):int(y1f * H), int(x0f * W):int(x1f * W)]
                crop = cv2.resize(crop, None, fx=args.upscale, fy=args.upscale,
                                  interpolation=cv2.INTER_CUBIC)
                res = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                    int(idx / fps * 1000))
                collected.append((idx, list(res.pose_landmarks or [])))
                frames[idx] = crop
                idx += 1
        cap.release()
        if not collected:
            continue

        n_frames = len(collected)
        tracks = build_tracks(collected, args.match_radius)
        keep = [(ti, t) for ti, t in enumerate(tracks)
                if len(t["frames"]) >= args.min_coverage * n_frames]
        if not keep:
            continue
        ch, cw = frames[collected[0][0]].shape[:2]

        # Montage: three frames spread over the retained window, every track in a stable colour.
        picks = [collected[int(k)][0] for k in np.linspace(0, n_frames - 1, 3)]
        panels = []
        for fi in picks:
            ann = frames[fi].copy()
            for slot, (ti, t) in enumerate(keep):
                lms = t["landmarks"].get(fi)
                if lms is None:
                    continue
                col = COLORS[slot % len(COLORS)]
                for l in lms:
                    cv2.circle(ann, (int(l.x * cw), int(l.y * ch)), 4, col, -1)
                bx0 = int(min(l.x for l in lms) * cw); by0 = int(min(l.y for l in lms) * ch)
                bx1 = int(max(l.x for l in lms) * cw); by1 = int(max(l.y for l in lms) * ch)
                cv2.rectangle(ann, (bx0, by0), (bx1, by1), col, 3)
                cv2.putText(ann, f"T{slot}", (bx0, max(by0 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
            panels.append(cv2.resize(ann, (cw // 2, ch // 2)))
        cv2.imwrite(str(OUT_DIR / f"{clip['clip_id']}_tracks.png"), np.hstack(panels))

        for slot, (ti, t) in enumerate(keep):
            f = track_features(t, n_frames, ch)
            rows.append({"clip_id": clip["clip_id"], "outcome": clip["outcome"],
                         "track_slot": slot, "n_frames": n_frames, **f})
        print(f"  [{ci}/{len(clips)}] {clip['clip_id']} {clip['outcome']:12s} "
              f"{len(keep)} candidate tracks of {len(tracks)}")

    with (OUT_DIR / "candidates.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    labels_path = OUT_DIR / "labels.csv"
    if not labels_path.exists():
        with labels_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["clip_id", "batter_track_slot", "note"])
            for cid in dict.fromkeys(r["clip_id"] for r in rows):
                w.writerow([cid, "", ""])
    print(f"\nmontages + candidates: {OUT_DIR.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"fill in batter_track_slot in {labels_path.name} (use -1 if the batter is not tracked)")


RULES = {
    "max_coverage": lambda ts: max(ts, key=lambda t: t["coverage"]),
    "max_wrist_speed": lambda ts: max(ts, key=lambda t: t["wrist_speed"]),
    "max_wrist_range": lambda ts: max(ts, key=lambda t: t["wrist_range"]),
    "max_bbox_height": lambda ts: max(ts, key=lambda t: t["bbox_height"]),
    "max_centroid_travel": lambda ts: max(ts, key=lambda t: t["centroid_travel"]),
    "max_wrist_range_excl_bottom": lambda ts: max(
        [t for t in ts if t["centroid_y"] <= 0.75] or ts, key=lambda t: t["wrist_range"]),
    "max_coverage_excl_bottom": lambda ts: max(
        [t for t in ts if t["centroid_y"] <= 0.75] or ts, key=lambda t: t["coverage"]),
}


def score(args) -> None:
    cand = list(csv.DictReader((OUT_DIR / "candidates.csv").open()))
    labels = {r["clip_id"]: r["batter_track_slot"]
              for r in csv.DictReader((OUT_DIR / "labels.csv").open())
              if r["batter_track_slot"].strip()}
    if not labels:
        raise SystemExit(f"no labels filled in yet: {OUT_DIR / 'labels.csv'}")

    by_clip: dict[str, list[dict]] = {}
    for r in cand:
        rec = {k: (float(v) if k not in ("clip_id", "outcome") else v) for k, v in r.items()}
        by_clip.setdefault(r["clip_id"], []).append(rec)

    labelled = {c: int(v) for c, v in labels.items() if c in by_clip}
    tracked = {c: v for c, v in labelled.items() if v >= 0}
    print(f"labelled clips: {len(labelled)}   batter tracked in {len(tracked)}   "
          f"not tracked in {len(labelled) - len(tracked)}")
    print(f"\n{'rule':34s} {'correct':>10s} {'accuracy':>9s}")
    results = []
    for name, rule in RULES.items():
        ok = 0
        for cid, slot in tracked.items():
            pick = rule(by_clip[cid])
            if int(pick["track_slot"]) == slot:
                ok += 1
        acc = ok / len(tracked) if tracked else float("nan")
        results.append({"rule": name, "correct": ok, "n": len(tracked), "accuracy": acc})
        print(f"{name:34s} {ok:>4d}/{len(tracked):<5d} {acc:>9.3f}")
    with (OUT_DIR / "rule_scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nbest rule: {best['rule']} at {best['accuracy']:.3f}")
    print("A rule is only worth shipping if it also holds on clips outside this labelled set.")


def main() -> None:
    args = parse_args()
    (render if args.mode == "render" else score)(args)


if __name__ == "__main__":
    main()
